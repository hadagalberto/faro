using Faro.Filtering;
using Faro.Hybrid;
using Faro.Payload;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Collection;

public class FaroCollectionTests
{
    private const int Dimensions = 16;
    private const int Count = 10;

    private static float[] OneHot(int index)
    {
        var vector = new float[Dimensions];
        vector[index] = 1f;
        return vector;
    }

    private static FaroCollection<Doc> Create(SearchPlanner? planner = null, PayloadSchema<Doc>? schema = null)
    {
        var options = new CollectionOptions
        {
            Planner = planner ?? new SearchPlanner(),
            DefaultEfSearch = 64,
        };
        return new FaroCollection<Doc>(
            schema ?? TestFixtures.Schema(),
            TestFixtures.Serializer(),
            new HnswOptions
            {
                Dimensions = Dimensions,
                Distance = Distance.DotProduct,
                M = 8,
                EfConstruction = 100,
                EfSearch = 64,
                Seed = 0x1234,
            },
            options);
    }

    private static FaroCollection<Doc> Populate(FaroCollection<Doc> collection)
    {
        for (int i = 0; i < Count; i++)
        {
            var doc = new Doc
            {
                Tenant = i % 5 == 0 ? "acme" : "globex",
                Year = 2019 + i,
                Active = i % 2 == 0,
                Tags = [i % 3 == 0 ? "red" : "blue"],
                Text = i switch
                {
                    2 => "gato",
                    3 => "cachorro",
                    4 => "gato cachorro",
                    7 => "gato gato gato",
                    _ => "outros assuntos",
                },
            };
            Assert.Equal(i, collection.Add(OneHot(i), doc));
        }

        return collection;
    }

    [Fact]
    public void Add_TryGetPayload_RoundTripsAllFields()
    {
        FaroCollection<Doc> collection = Populate(Create());

        Assert.Equal(Count, collection.Count);
        Assert.True(collection.TryGetPayload(3, out Doc? payload));
        Assert.NotNull(payload);
        Assert.Equal("globex", payload.Tenant);
        Assert.Equal(2022, payload.Year);
        Assert.False(payload.Active);
        Assert.Equal(["red"], payload.Tags);
        Assert.Equal("cachorro", payload.Text);
        Assert.False(collection.TryGetPayload(999, out _));
    }

    [Fact]
    public void UnfilteredSearch_ReturnsNearestBestFirst()
    {
        FaroCollection<Doc> collection = Populate(Create());

        SearchResult[] hits = collection.Search(OneHot(3), k: 3);

        Assert.Equal(3, hits[0].Id);
        Assert.Equal(1f, hits[0].Score, 6);
        Assert.All(hits, hit => Assert.True(hit.Score >= 0f));
    }

    [Fact]
    public void SelectiveFilter_PlansExactCandidatesAndReturnsOnlyMatches()
    {
        // A 0.25 threshold routes the 2-of-10 (20%) filter through the exact branch; the 1%
        // selectivity gate itself is exercised in Faro.RecallTests.
        FaroCollection<Doc> collection = Populate(Create(new SearchPlanner(0.25, 4096)));
        Filter filter = Filter.Eq("tenant", "acme");

        SearchResult[] hits = collection.Search(OneHot(0), k: 10, filter);

        Assert.Equal(SearchStrategy.ExactCandidates, collection.LastDecision.Strategy);
        Assert.Equal(2, collection.LastDecision.CandidateCount);
        Assert.Equal(0.2, collection.LastDecision.Selectivity, 6);
        Assert.Equal([0, 5], hits.Select(hit => hit.Id).ToArray());
        Assert.DoesNotContain(hits, hit => hit.Id is not (0 or 5));
    }

    [Fact]
    public void BroadFilter_PlansFilteredTraversalWithAdaptiveEf()
    {
        var planner = new SearchPlanner(exactThreshold: 0.01, maxEf: 4096);
        FaroCollection<Doc> collection = Populate(Create(planner));
        Filter filter = Filter.Eq("tenant", "acme");

        SearchResult[] hits = collection.Search(OneHot(0), k: 3, filter, efSearch: 32);

        Assert.Equal(SearchStrategy.FilteredTraversal, collection.LastDecision.Strategy);
        Assert.Equal(160, collection.LastDecision.EffectiveEf); // ceil(32 / 0.2)
        Assert.All(hits, hit => Assert.True(hit.Id is 0 or 5));
    }

    [Fact]
    public void CompositeFilter_AndOrNot_AreHonoured()
    {
        FaroCollection<Doc> collection = Populate(Create());

        Filter activeRecent = Filter.Eq("active", true) & Filter.Range("year", 2022, 2026);
        SearchResult[] hits = collection.Search(OneHot(8), k: 10, activeRecent);
        Assert.Equal([4, 6], hits.Select(hit => hit.Id).OrderBy(id => id).ToArray());

        Filter union = Filter.Eq("tenant", "acme") | Filter.In("tags", "red");
        SearchResult[] unionHits = collection.Search(OneHot(0), k: 10, union);
        Assert.Equal([0, 3, 5, 6, 9], unionHits.Select(hit => hit.Id).OrderBy(id => id).ToArray());

        SearchResult[] negated = collection.Search(OneHot(1), k: 10, !Filter.Eq("tenant", "acme"));
        Assert.DoesNotContain(negated, hit => hit.Id is 0 or 5);
    }

    [Fact]
    public void Delete_RemovesPayloadIndexTextAndVector()
    {
        FaroCollection<Doc> collection = Populate(Create());

        Assert.True(collection.Delete(0));
        Assert.False(collection.Delete(0));
        Assert.Equal(Count - 1, collection.Count);
        Assert.False(collection.TryGetPayload(0, out _));

        SearchResult[] hits = collection.Search(OneHot(0), k: 10, Filter.Eq("tenant", "acme"));
        Assert.Equal([5], hits.Select(hit => hit.Id).ToArray());
        Assert.True(collection.TextIndex.DocumentCount == Count - 1);

        collection.Compact();
        Assert.Equal(Count - 1, collection.Count);
    }

    [Fact]
    public void UnknownOrTextFilterField_Throws()
    {
        FaroCollection<Doc> collection = Populate(Create());

        Assert.Throws<ArgumentException>(() => collection.Search(OneHot(0), 3, Filter.Eq("unknown", "x")));
        Assert.Throws<ArgumentException>(() => collection.Search(OneHot(0), 3, Filter.Range("text", 1, 2)));
    }

    [Fact]
    public void HybridRrf_FusesBothRankings()
    {
        FaroCollection<Doc> collection = Populate(Create());

        HybridResult[] hits = collection.HybridSearch(OneHot(3), "gato", k: 6);

        Assert.NotEmpty(hits);
        Assert.Equal(1, hits.First(hit => hit.Id == 3).VectorRank);
        Assert.Contains(hits, hit => hit.Id == 2 && hit.TextRank > 0);
        Assert.Contains(hits, hit => hit.Id == 4 && hit.TextRank > 0);
        Assert.Contains(hits, hit => hit.Id == 1 && hit.VectorRank > 0 && hit.TextRank == 0);
        Assert.True(hits[0].Score > 0);
        Assert.Equal(hits.Length, hits.Select(hit => hit.Id).Distinct().Count());
    }

    [Fact]
    public void HybridWeighted_TextHeavyWeightsReorderResults()
    {
        FaroCollection<Doc> collection = Populate(Create());

        HybridResult[] hits = collection.HybridSearch(
            OneHot(0),
            "gato",
            k: 10,
            fusion: Fusion.Weighted(0.1, 0.9));

        Assert.Equal(7, hits[0].Id); // three "gato" terms beat the vector-perfect doc 0
        Assert.Equal(1, hits.First(hit => hit.Id == 0).VectorRank);
    }

    [Fact]
    public void HybridSearch_AccentFoldedQueryFindsAccentedText()
    {
        FaroCollection<Doc> collection = Create();
        collection.Add(OneHot(0), new Doc { Tenant = "x", Text = "ação na árvore" });

        HybridResult[] hits = collection.HybridSearch(OneHot(0), "acao", k: 3);

        Assert.Equal(1, hits[0].TextRank);
        Assert.True(hits[0].TextScore > 0);
    }

    [Fact]
    public void HybridSearch_RespectsFilterOnBothLegs()
    {
        FaroCollection<Doc> collection = Populate(Create());

        HybridResult[] hits = collection.HybridSearch(
            OneHot(3),
            "gato",
            k: 10,
            filter: Filter.Eq("tenant", "acme"));

        Assert.All(hits, hit => Assert.True(hit.Id is 0 or 5));
    }

    [Fact]
    public void HybridSearch_WithoutFullText_Throws()
    {
        var schema = new PayloadSchema<Doc>().Keyword("tenant", d => d.Tenant);
        FaroCollection<Doc> collection = Create(schema: schema);

        Assert.Throws<InvalidOperationException>(() => collection.HybridSearch(OneHot(0), "x", 3));
    }

    [Fact]
    public void InvalidArguments_AreRejected()
    {
        FaroCollection<Doc> collection = Populate(Create());

        Assert.Throws<ArgumentOutOfRangeException>(() => collection.Search(OneHot(0), 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => collection.Search(OneHot(0), 3, efSearch: 0));
        Assert.Throws<ArgumentException>(() => collection.Search(new float[Dimensions + 1], 3));
        Assert.Throws<ArgumentNullException>(() => collection.HybridSearch(OneHot(0), null!, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => collection.HybridSearch(OneHot(0), "x", 0));
    }
}
