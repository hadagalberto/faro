using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.VectorData;

namespace Faro.VectorData.Tests;

public sealed class Article
{
    [VectorStoreKey]
    public string Id { get; set; } = string.Empty;

    [VectorStoreData(IsIndexed = true)]
    public string Tenant { get; set; } = string.Empty;

    [VectorStoreData(IsIndexed = true)]
    public int Year { get; set; }

    [VectorStoreData(IsIndexed = true)]
    public bool Published { get; set; }

    [VectorStoreData(IsIndexed = true)]
    public string[] Tags { get; set; } = [];

    [VectorStoreData(IsFullTextIndexed = true)]
    public string Text { get; set; } = string.Empty;

    [VectorStoreVector(4, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public float[] Vector { get; set; } = [];
}

[JsonSerializable(typeof(Article))]
[JsonSerializable(typeof(Dictionary<string, object?>))]
[JsonSerializable(typeof(object))]
[JsonSerializable(typeof(float[]))]
internal sealed partial class VectorDataJsonContext : JsonSerializerContext;

internal sealed class DictionaryPayloadSerializer :
    Faro.Payload.IPayloadSerializer<Dictionary<string, object?>>
{
    public byte[] Serialize(Dictionary<string, object?> value) =>
        JsonSerializer.SerializeToUtf8Bytes(value, VectorDataJsonContext.Default.DictionaryStringObject);

    public Dictionary<string, object?>? Deserialize(ReadOnlySpan<byte> utf8Json) =>
        JsonSerializer.Deserialize(utf8Json, VectorDataJsonContext.Default.DictionaryStringObject);
}

internal static class Fixtures
{
    public static FaroVectorStoreOptions Options() =>
        new FaroVectorStoreOptions
        {
            JsonSerializerContext = VectorDataJsonContext.Default,
            Index = new HnswOptions { M = 8, EfConstruction = 80, EfSearch = 64, Seed = 12345 },
        };

    public static Article Article(
        string id, string tenant, int year, bool published, string[] tags, string text, float[] vector) =>
        new()
        {
            Id = id,
            Tenant = tenant,
            Year = year,
            Published = published,
            Tags = tags,
            Text = text,
            Vector = vector,
        };

    public static async Task FillAsync(VectorStoreCollection<string, Article> collection)
    {
        await collection.UpsertAsync(
        [
            Article("a", "acme", 2021, true, ["ai", "search"],
                "hnsw graph index makes vector search fast", [1, 0, 0, 0]),
            Article("b", "acme", 2023, false, ["search", "db"],
                "bm25 ranks full text results by term frequency", [0, 1, 0, 0]),
            Article("c", "other", 2019, true, ["ai"],
                "quantization compresses embeddings into bytes", [0.9f, 0.1f, 0, 0]),
            Article("d", "other", 2024, true, ["db"],
                "persistence uses a write ahead log for durability", [-0.1f, 0.995f, 0, 0]),
        ]).ConfigureAwait(false);
    }
}
