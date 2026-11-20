using Microsoft.Extensions.VectorData;
using Xunit;

namespace Faro.VectorData.Tests;

public sealed class FilterTranslationTests
{
    [Fact]
    public async Task EqualityAndConjunction_selectExactRecords()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        Article[] matches = await GetAsync(collection, article =>
            article.Tenant == "acme" && article.Year >= 2021);

        Assert.Equal(["a", "b"], matches.Select(article => article.Id).OrderBy(id => id).ToArray());
    }

    [Fact]
    public async Task CapturedVariable_isInlinedAndTranslated()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);
        const string tenant = "other";

        Article[] matches = await GetAsync(collection, article => article.Tenant == tenant);

        Assert.Equal(["c", "d"], matches.Select(article => article.Id).OrderBy(id => id).ToArray());
    }

    [Fact]
    public async Task DisjunctionNegationAndInequality_areTranslated()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        Article[] orMatches = await GetAsync(collection, article => article.Year == 2021 || article.Year == 2024);
        Assert.Equal(["a", "d"], orMatches.Select(article => article.Id).OrderBy(id => id).ToArray());

        Article[] notMatches = await GetAsync(collection, article => !(article.Tenant == "acme"));
        Assert.Equal(["c", "d"], notMatches.Select(article => article.Id).OrderBy(id => id).ToArray());

        Article[] notEqual = await GetAsync(collection, article => article.Tenant != "acme");
        Assert.Equal(["c", "d"], notEqual.Select(article => article.Id).OrderBy(id => id).ToArray());
    }

    [Fact]
    public async Task BoolAndNumericRanges_areTranslated()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        Article[] published = await GetAsync(collection, article => article.Published);
        Assert.Equal(["a", "c", "d"], published.Select(article => article.Id).OrderBy(id => id).ToArray());

        Article[] range = await GetAsync(collection, article => article.Year > 2019 && article.Year <= 2023);
        Assert.Equal(["a", "b"], range.Select(article => article.Id).OrderBy(id => id).ToArray());
    }

    [Fact]
    public async Task TagContains_translatesInEitherDirection()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        Article[] memberContains = await GetAsync(collection, article => article.Tags.Contains("ai"));
        Assert.Equal(["a", "c"], memberContains.Select(article => article.Id).OrderBy(id => id).ToArray());

        Article[] constantContains = await GetAsync(collection, article => new[] { "acme" }.Contains(article.Tenant));
        Assert.Equal(["a", "b"], constantContains.Select(article => article.Id).OrderBy(id => id).ToArray());
    }

    [Fact]
    public async Task Filter_appliesToVectorSearch()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        var options = new VectorSearchOptions<Article> { Filter = article => article.Tenant == "other" };
        var hits = new List<VectorSearchResult<Article>>();
        await foreach (VectorSearchResult<Article> hit in collection.SearchAsync(
                           new float[] { 1, 0, 0, 0 }, 5, options))
        {
            hits.Add(hit);
        }

        Assert.Equal(["c", "d"], hits.Select(hit => hit.Record.Id).OrderBy(id => id).ToArray());
    }

    [Fact]
    public async Task OrderBy_sortsFilteredRecords()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        var options = new FilteredRecordRetrievalOptions<Article>
        {
            OrderBy = definition => definition.Descending(article => article.Year),
        };
        var results = new List<Article>();
        await foreach (Article article in collection.GetAsync(_ => true, 10, options))
        {
            results.Add(article);
        }

        Assert.Equal([2024, 2023, 2021, 2019], results.Select(article => article.Year).ToArray());
    }

    [Fact]
    public async Task UnsupportedExpressions_failLoudly()
    {
        using var store = FaroVectorStore.InMemory(Fixtures.Options());
        var collection = store.GetCollection<string, Article>("articles");
        await Fixtures.FillAsync(collection);

        await Assert.ThrowsAsync<NotSupportedException>(async () =>
            await GetAsync(collection, article => article.Text.Contains("hnsw")));
        await Assert.ThrowsAsync<NotSupportedException>(async () =>
            await GetAsync(collection, article => article.Id.StartsWith("a")));
        await Assert.ThrowsAsync<NotSupportedException>(async () =>
            await GetAsync(collection, article => article.Year + 1 == 2022));
    }

    private static async Task<Article[]> GetAsync(
        VectorStoreCollection<string, Article> collection,
        System.Linq.Expressions.Expression<Func<Article, bool>> filter)
    {
        var results = new List<Article>();
        await foreach (Article article in collection.GetAsync(filter, 20))
        {
            results.Add(article);
        }

        return results.ToArray();
    }
}
