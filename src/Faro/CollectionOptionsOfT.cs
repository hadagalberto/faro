using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using Faro.Payload;

namespace Faro;

/// <summary>
/// Definition and tuning of a collection inside a <see cref="FaroDb"/>.
/// </summary>
/// <remarks>
/// Payload bytes are always produced with an AOT-safe source-generated converter. Supply either a
/// <see cref="Serializer"/>, a <see cref="TypeInfo"/>, or a <see cref="Context"/> that knows
/// <typeparamref name="TPayload"/>. No reflection-based JSON path exists in this library.
/// </remarks>
/// <typeparam name="TPayload">The payload type stored in the collection.</typeparam>
public sealed class CollectionOptions<TPayload> : CollectionOptions
    where TPayload : class
{
    /// <summary>Number of dimensions of every vector in the collection. Required.</summary>
    public required int Dimensions { get; init; }

    /// <summary>Distance metric. Defaults to <see cref="Distance.Cosine"/>.</summary>
    public Distance Distance { get; init; } = Distance.Cosine;

    /// <summary>HNSW construction and search tuning. Dimensions are taken from <see cref="Dimensions"/>.</summary>
    public HnswOptions? Index { get; init; }

    /// <summary>Optional quantized search configuration.</summary>
    public QuantizationOptions? Quantization { get; init; }

    /// <summary>
    /// Selector for the full-text field fed into BM25. A schema can also be supplied explicitly
    /// through <see cref="Schema"/> when filterable fields are needed.
    /// </summary>
    public Func<TPayload, string?>? FullTextField { get; init; }

    /// <summary>Declared payload fields. When omitted, a single full-text field is derived from <see cref="FullTextField"/>.</summary>
    public PayloadSchema<TPayload>? Schema { get; init; }

    /// <summary>Prebuilt source-generated payload serializer.</summary>
    public IPayloadSerializer<TPayload>? Serializer { get; init; }

    /// <summary>Source-generated type information for <typeparamref name="TPayload"/>.</summary>
    public JsonTypeInfo<TPayload>? TypeInfo { get; init; }

    /// <summary>A generated JSON context containing <typeparamref name="TPayload"/>.</summary>
    public JsonSerializerContext? Context { get; init; }

    internal HnswOptions BuildIndexOptions()
    {
        if (Dimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(Dimensions), Dimensions, "Dimensions must be positive.");
        }

        var index = new HnswOptions
        {
            Dimensions = Dimensions,
            Distance = Distance,
        };

        if (Index is not null)
        {
            index.M = Index.M;
            index.EfConstruction = Index.EfConstruction;
            index.EfSearch = Index.EfSearch;
            index.NormalizeOnAdd = Index.NormalizeOnAdd;
            index.LevelMultiplier = Index.LevelMultiplier;
            index.Seed = Index.Seed;
            index.ExtendCandidates = Index.ExtendCandidates;
            index.KeepPrunedConnections = Index.KeepPrunedConnections;
            index.Mc = Index.Mc;
            index.InitialCapacity = Index.InitialCapacity;
        }

        index.Quantization = Quantization;
        // HnswIndex validates every tunable, including M, ef and the metric/quantization match.
        _ = new HnswIndex(index);
        return index;
    }

    internal PayloadSchema<TPayload> BuildSchema()
    {
        if (Schema is not null)
        {
            return Schema;
        }

        if (FullTextField is null)
        {
            throw new ArgumentException(
                "a collection needs either a Schema or a FullTextField selector", nameof(Schema));
        }

        return new PayloadSchema<TPayload>().FullText("text", FullTextField);
    }

    internal IPayloadSerializer<TPayload> ResolveSerializer()
    {
        if (Serializer is not null)
        {
            return Serializer;
        }

        if (TypeInfo is not null)
        {
            return new SourceGenPayloadSerializer<TPayload>(TypeInfo);
        }

        if (Context is not null && Context.GetTypeInfo(typeof(TPayload)) is JsonTypeInfo<TPayload> info)
        {
            return new SourceGenPayloadSerializer<TPayload>(info);
        }

        throw new ArgumentException(
            "payload persistence requires Serializer, TypeInfo or Context with source-generated metadata",
            nameof(Serializer));
    }
}
