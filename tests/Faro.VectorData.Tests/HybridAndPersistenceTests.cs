using Microsoft.Extensions.VectorData;
using Xunit;

namespace Faro.VectorData.Tests;

public sealed class HybridAndPersistenceTests
{
    [Fact]
    public async Task HybridSearch_fusesVectorAndBm25Scores()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        var hybrid = (IKeywordHybridSearchable<Article>)collection;
        VectorSearchResult<Article>[] hits = await ToArrayAsync(hybrid.HybridSearchAsync(
            new float[] { 0, 1, 0, 0 }, ["bm25"], 4));

        Assert.NotEmpty(hits);
        Assert.Equal("b", hits[0].Record.Id);
        Assert.All(hits, hit => Assert.True(hit.Score > 0));
    }

    [Fact]
    public async Task HybridSearch_appliesFilterToBothLegs()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        var options = new HybridSearchOptions<Article>
        {
            Filter = article => article.Tenant == "acme",
            AdditionalProperty = article => article.Text,
        };
        var hybrid = (IKeywordHybridSearchable<Article>)collection;
        VectorSearchResult<Article>[] hits = await ToArrayAsync(hybrid.HybridSearchAsync(
            new float[] { 0, 1, 0, 0 }, ["hnsw", "graph"], 5, options));

        Assert.NotEmpty(hits);
        Assert.All(hits, hit => Assert.Equal("acme", hit.Record.Tenant));
        Assert.Equal("a", hits[0].Record.Id);
    }

    [Fact]
    public async Task HybridSearch_rejectsEmptyKeywords()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);
        var hybrid = (IKeywordHybridSearchable<Article>)collection;

        await Assert.ThrowsAsync<ArgumentException>(async () =>
            await ToArrayAsync(hybrid.HybridSearchAsync(new float[] { 1, 0, 0, 0 }, [], 3)));
    }

    [Fact]
    public async Task FileBackedStore_survivesReopen()
    {
        string directory = Path.Combine(Path.GetTempPath(), "faro-vd-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "vector.faro");
        try
        {
            using (var store = new FaroVectorStore(path, options: Fixtures.Options()))
            {
                var collection = store.GetCollection<string, Article>("articles");
                await Fixtures.FillAsync(collection);
            }

            using var reopened = new FaroVectorStore(path, options: Fixtures.Options());
            var loaded = reopened.GetCollection<string, Article>("articles");
            Article? record = await loaded.GetAsync("c");
            Assert.NotNull(record);
            Assert.Equal("quantization compresses embeddings into bytes", record.Text);

            VectorSearchResult<Article>[] hits = await ToArrayAsync(
                loaded.SearchAsync(new float[] { 1, 0, 0, 0 }, 2));
            Assert.NotEmpty(hits);
            Assert.Equal("a", hits[0].Record.Id);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static async Task<T[]> ToArrayAsync<T>(IAsyncEnumerable<T> source)
    {
        var results = new List<T>();
        await foreach (T item in source)
        {
            results.Add(item);
        }

        return results.ToArray();
    }
}
