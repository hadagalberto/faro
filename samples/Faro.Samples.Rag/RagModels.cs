using System.Text.Json.Serialization;

namespace Faro.Samples.Rag;

/// <summary>Payload stored for one markdown chunk.</summary>
public sealed class Chunk
{
    public string Title { get; set; } = string.Empty;

    public string Text { get; set; } = string.Empty;
}

/// <summary>One hybrid retrieval result returned by <see cref="RagPipeline"/>.</summary>
public readonly record struct RagSearchResult(string Title, string Text, double Score, double VectorScore, double TextScore);

/// <summary>Source-generated payload metadata; no reflection is used when persisting chunks.</summary>
[JsonSerializable(typeof(Chunk))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class RagJsonContext : JsonSerializerContext;
