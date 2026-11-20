using Microsoft.Extensions.VectorData;
using Xunit;

namespace Faro.VectorData.Tests;

public sealed class CrudAndSearchTests
{
    [Fact]
    public async Task UpsertGetDelete_roundTripsKnownRecords()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");

        await collection.UpsertAsync(
            Fixtures.Article("one", "acme", 2020, true, ["ai"], "hnsw vectors", [1, 0, 0, 0]));

        Article? loaded = await collection.GetAsync("one");
        Assert.NotNull(loaded);
        Assert.Equal("acme", loaded.Tenant);
        Assert.Equal(2020, loaded.Year);
        Assert.True(loaded.Published);
        Assert.Equal(["ai"], loaded.Tags);
        Assert.Equal([1f, 0f, 0f, 0f], loaded.Vector);

        await collection.DeleteAsync("one");
        Assert.Null(await collection.GetAsync("one"));
    }

    [Fact]
    public async Task BatchUpsert_isIdempotentForSameKey()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");

        await collection.UpsertAsync(
        [
            Fixtures.Article("a", "acme", 2020, true, ["a"], "first text", [1, 0, 0, 0]),
            Fixtures.Article("a", "acme", 2021, false, ["b"], "replacement text", [0, 1, 0, 0]),
        ]);

        Article? loaded = await collection.GetAsync("a");
        Assert.NotNull(loaded);
        Assert.Equal(2021, loaded.Year);
        Assert.Equal("replacement text", loaded.Text);
    }

    [Fact]
    public async Task Search_returnsKnownNearestNeighboursFirst()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        VectorSearchResult<Article>[] results = await ToArrayAsync(collection.SearchAsync(
            new float[] { 1, 0, 0, 0 }, 3));

        Assert.Equal(3, results.Length);
        Assert.Equal("a", results[0].Record.Id);
        Assert.Equal("c", results[1].Record.Id);
        Assert.Equal("b", results[2].Record.Id);
        Assert.InRange(results[0].Score!.Value, 0.999, 1.001);
        Assert.True(results[0].Score > results[1].Score);
        Assert.True(results[1].Score > results[2].Score);
    }

    [Fact]
    public async Task Search_honoursSkipScoreThresholdAndIncludeVectors()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        var options = new VectorSearchOptions<Article>
        {
            Skip = 1,
            IncludeVectors = false,
            ScoreThreshold = 0.5,
        };
        VectorSearchResult<Article>[] results = await ToArrayAsync(
            collection.SearchAsync(new float[] { 1, 0, 0, 0 }, 3, options));

        Assert.Single(results);
        Assert.Equal("c", results[0].Record.Id);
        Assert.All(results, result => Assert.Empty(result.Record.Vector));
        Assert.All(results, result => Assert.True(result.Score >= 0.5));
    }

    [Fact]
    public async Task Search_withScoreThresholdFiltersLowSimilarity()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        var options = new VectorSearchOptions<Article> { ScoreThreshold = 0.9 };
        VectorSearchResult<Article>[] results = await ToArrayAsync(
            collection.SearchAsync(new float[] { 1, 0, 0, 0 }, 4, options));

        Assert.Equal(["a", "c"], results.Select(result => result.Record.Id).OrderBy(id => id).ToArray());
    }

    [Fact]
    public async Task Search_rejectsWrongDimensionalityWithVectorStoreException()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        await Assert.ThrowsAnyAsync<ArgumentException>(async () =>
            await ToArrayAsync(collection.SearchAsync(new float[] { 1, 2, 3 }, 1)));
    }

    [Fact]
    public async Task CollectionMetadata_andLifecycleAreExposed()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");

        Assert.Equal("articles", collection.Name);
        Assert.True(await store.CollectionExistsAsync("articles"));
        await collection.EnsureCollectionExistsAsync();
        Assert.True(await store.CollectionExistsAsync("articles"));
        Assert.Contains("articles", await ListAsync(store.ListCollectionNamesAsync()));

        var metadata = Assert.IsType<VectorStoreCollectionMetadata>(collection.GetService(typeof(VectorStoreCollectionMetadata)));
        Assert.Equal("faro", metadata.VectorStoreSystemName);
        Assert.Equal("articles", metadata.CollectionName);

        await collection.EnsureCollectionDeletedAsync();
        Assert.False(await store.CollectionExistsAsync("articles"));
        await Assert.ThrowsAsync<VectorStoreException>(async () => await collection.GetAsync("a"));
    }

    [Fact]
    public async Task DynamicCollection_supportsExplicitDefinitionAndStringKeys()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(string)),
                new VectorStoreDataProperty("Tenant", typeof(string)) { IsIndexed = true },
                new VectorStoreDataProperty("Text", typeof(string)) { IsFullTextIndexed = true },
                new VectorStoreVectorProperty("Vector", typeof(float[]), 4)
                {
                    DistanceFunction = DistanceFunction.CosineSimilarity,
                },
            ],
        };
        var collection = store.GetDynamicCollection("dynamic", definition);

        var record = new Dictionary<string, object?>
        {
            ["Id"] = "k1",
            ["Tenant"] = "acme",
            ["Text"] = "hnsw graph search",
            ["Vector"] = new float[] { 1, 0, 0, 0 },
        };
        await collection.UpsertAsync(record);

        Dictionary<string, object?>? loaded = await collection.GetAsync("k1");
        Assert.NotNull(loaded);
        Assert.Equal("acme", loaded["Tenant"]);

        VectorSearchResult<Dictionary<string, object?>>[] hits = await ToArrayAsync(
            collection.SearchAsync(new float[] { 0.9f, 0.1f, 0, 0 }, 1));
        Assert.Equal("k1", hits[0].Record["Id"]);
    }

    private static async Task<T[]> ToArrayAsync<T>(IAsyncEnumerable<T> source)
    {
        var items = new List<T>();
        await foreach (T item in source)
        {
            items.Add(item);
        }

        return items.ToArray();
    }

    private static async Task<T[]> ListAsync<T>(IAsyncEnumerable<T> source) => await ToArrayAsync(source);
}
