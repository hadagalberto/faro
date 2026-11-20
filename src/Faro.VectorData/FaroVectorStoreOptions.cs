using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using Faro.Payload;
using Microsoft.Extensions.AI;

namespace Faro.VectorData;

/// <summary>
/// Configuration shared by every collection obtained from a <see cref="FaroVectorStore"/>.
/// </summary>
/// <remarks>
/// Faro persists payloads with source-generated <see cref="System.Text.Json"/> metadata only, so a
/// serializer for each record type must be supplied. Either register a
/// <see cref="JsonSerializerContext"/> that knows all record types, or register individual
/// <see cref="IPayloadSerializer{TRecord}"/> instances.
/// </remarks>
public sealed class FaroVectorStoreOptions
{
    private readonly Dictionary<Type, object> _serializers = [];

    /// <summary>A source-generated JSON context used to resolve payload serializers on demand.</summary>
    public JsonSerializerContext? JsonSerializerContext { get; set; }

    /// <summary>Optional default embedding generator applied to vector properties.</summary>
    public IEmbeddingGenerator? EmbeddingGenerator { get; set; }

    /// <summary>HNSW tuning applied to created collections.</summary>
    public HnswOptions? Index { get; set; }

    /// <summary>Optional quantization applied to created collections.</summary>
    public QuantizationOptions? Quantization { get; set; }

    /// <summary>Default search beam width applied when a search does not override it.</summary>
    public int? DefaultEfSearch { get; set; }

    /// <summary>Registers an explicit AOT-safe serializer for one record type.</summary>
    /// <typeparam name="TRecord">The record type.</typeparam>
    /// <param name="serializer">The serializer to use.</param>
    /// <returns>This options instance.</returns>
    public FaroVectorStoreOptions RegisterPayloadSerializer<TRecord>(IPayloadSerializer<TRecord> serializer)
        where TRecord : class
    {
        ArgumentNullException.ThrowIfNull(serializer);
        _serializers[typeof(TRecord)] = serializer;
        return this;
    }

    internal IPayloadSerializer<TRecord>? ResolveSerializer<TRecord>()
        where TRecord : class
    {
        if (_serializers.TryGetValue(typeof(TRecord), out object? registered) &&
            registered is IPayloadSerializer<TRecord> typed)
        {
            return typed;
        }

        if (JsonSerializerContext?.GetTypeInfo(typeof(TRecord)) is JsonTypeInfo<TRecord> info)
        {
            return new SourceGenPayloadSerializer<TRecord>(info);
        }

        return null;
    }

    internal FaroVectorStoreOptions Clone()
    {
        var clone = new FaroVectorStoreOptions
        {
            JsonSerializerContext = JsonSerializerContext,
            EmbeddingGenerator = EmbeddingGenerator,
            Index = Index,
            Quantization = Quantization,
            DefaultEfSearch = DefaultEfSearch,
        };
        foreach ((Type type, object serializer) in _serializers)
        {
            clone._serializers[type] = serializer;
        }

        return clone;
    }
}
