using System.Diagnostics;
using Faro.Filtering;
using Faro.RecallTests.Shared;
using Faro.Testing;
using Xunit;

namespace Faro.RecallTests;

/// <summary>
/// Filtered recall gates at 1% selectivity. Ground truth is an exact FlatIndex over only the
/// matching vectors, so the measurement is independent of the HNSW graph and of the planner.
/// </summary>
public class FilteredRecallTests
{
    private const int Count = 10_000;
    private const int Dimensions = 32;
    private const int Queries = 20;
    private const int K = 10;
    private const int SelectivityEvery = 100; // 1% of points carry tenant = "match"

    private readonly ITestOutputHelper _output;

    public FilteredRecallTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void FilteredRecallAtOnePercentSelectivity_AtLeast09()
    {
        var data = new float[Count][];
        for (int i = 0; i < Count; i++)
        {
            data[i] = DeterministicRandom.GaussianVector(Dimensions, DeterministicRandom.DeriveSeed(0x5eed_f11d, i));
        }

        var queries = new float[Queries][];
        for (int q = 0; q < Queries; q++)
        {
            queries[q] = DeterministicRandom.GaussianVector(
                Dimensions,
                DeterministicRandom.DeriveSeed(0x5eed_1234, 100_000 + q));
        }

        var matchedIds = new List<int>();
        var flat = new FlatIndex(new FlatOptions
        {
            Dimensions = Dimensions,
            Distance = Distance.DotProduct,
            NormalizeOnAdd = false,
            ParallelThreshold = int.MaxValue,
        });
        for (int i = 0; i < Count; i++)
        {
            if (i % SelectivityEvery == 0)
            {
                flat.Add(data[i]);
                matchedIds.Add(i);
            }
        }

        Assert.Equal(Count / SelectivityEvery, matchedIds.Count);

        FaroCollection<RecallDoc> collection = BuildCollection();
        for (int i = 0; i < Count; i++)
        {
            collection.Add(data[i], new RecallDoc
            {
                Tenant = i % SelectivityEvery == 0 ? "match" : "other",
                Year = 2000 + (i % 25),
            });
        }

        Filter match = Filter.Eq("tenant", "match");
        var truths = new HashSet<int>[Queries];
        for (int q = 0; q < Queries; q++)
        {
            SearchResult[] exact = flat.Search(queries[q], K);
            truths[q] = exact.Select(hit => matchedIds[hit.Id]).ToHashSet();
            Assert.Equal(K, truths[q].Count);
        }

        _output.WriteLine("| strategy                         | N     | sel  | recall@10 | us/query |");
        _output.WriteLine("|----------------------------------|-------|------|-----------|----------|");

        double exactRecall = Measure(collection, queries, truths, match, "planner: exact candidates");
        Assert.True(exactRecall >= 0.9, $"exact-candidate filtered recall was {exactRecall:F3}");

        var traversalPlanner = new SearchPlanner(exactThreshold: 0.001, maxEf: 8192);
        SearchDecision decision = traversalPlanner.Plan(Count / SelectivityEvery, Count, 64);
        Assert.Equal(SearchStrategy.FilteredTraversal, decision.Strategy);
        Assert.Equal(6400, decision.EffectiveEf);

        collection.Planner = traversalPlanner;
        double seededRecall = Measure(collection, queries, truths, match, "planner: filtered traversal");
        Assert.True(seededRecall >= 0.9, $"filtered-traversal recall was {seededRecall:F3}");

        double unseededRecall = MeasureUnseeded(collection, queries, truths);
        Assert.True(unseededRecall >= 0.9, $"unseeded filtered-traversal recall was {unseededRecall:F3}");
    }

    private static FaroCollection<RecallDoc> BuildCollection() =>
        new(
            RecallFixtures.Schema(),
            RecallFixtures.Serializer(),
            new HnswOptions
            {
                Dimensions = Dimensions,
                Distance = Distance.DotProduct,
                NormalizeOnAdd = false,
                M = 16,
                EfConstruction = 200,
                EfSearch = 64,
                Seed = 0x5eed_f11d,
            },
            new CollectionOptions { DefaultEfSearch = 64 });

    private double Measure(
        FaroCollection<RecallDoc> collection,
        float[][] queries,
        HashSet<int>[] truths,
        Filter filter,
        string label)
    {
        double sum = 0;
        long elapsed = 0;
        for (int q = 0; q < Queries; q++)
        {
            var watch = Stopwatch.StartNew();
            SearchResult[] hits = collection.Search(queries[q], K, filter);
            watch.Stop();

            int found = hits.Count(hit => truths[q].Contains(hit.Id));
            sum += found / (double)K;
            elapsed += watch.ElapsedTicks;
        }

        double mean = sum / Queries;
        double micros = elapsed * 1_000_000.0 / Stopwatch.Frequency / Queries;
        _output.WriteLine($"| {label,-32} | {Count,5} | 1%   | {mean,9:F3} | {micros,8:F1} |");
        return mean;
    }

    private double MeasureUnseeded(
        FaroCollection<RecallDoc> collection,
        float[][] queries,
        HashSet<int>[] truths)
    {
        double sum = 0;
        long elapsed = 0;
        var watch = new Stopwatch();
        for (int q = 0; q < Queries; q++)
        {
            watch.Restart();
            SearchResult[] hits = collection.Index.SearchFiltered(
                queries[q],
                K,
                id => collection.TryGetPayload(id, out RecallDoc? doc) && doc!.Tenant == "match",
                efSearch: 6400);
            watch.Stop();

            int found = hits.Count(hit => truths[q].Contains(hit.Id));
            sum += found / (double)K;
            elapsed += watch.ElapsedTicks;
        }

        double mean = sum / Queries;
        double micros = elapsed * 1_000_000.0 / Stopwatch.Frequency / Queries;
        _output.WriteLine($"| {"planner: no seeds",-32} | {Count,5} | 1%   | {mean,9:F3} | {micros,8:F1} |");
        return mean;
    }
}
