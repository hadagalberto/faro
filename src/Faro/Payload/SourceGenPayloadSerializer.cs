using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace Faro.Payload;

/// <summary>
/// AOT-safe payload serializer backed by a source-generated <see cref="JsonTypeInfo{T}"/>.
/// </summary>
/// <remarks>
/// No reflection-based overload of <see cref="JsonSerializer"/> is used, so the payload path is
/// compatible with Native AOT and trimming. Callers obtain the type info from their own
/// <see cref="System.Text.Json.Serialization.JsonSerializerContext"/>, e.g. <c>MyContext.Default.Doc</c>.
/// </remarks>
/// <typeparam name="T">The payload type.</typeparam>
public sealed class SourceGenPayloadSerializer<T> : IPayloadSerializer<T>
    where T : class
{
    private readonly JsonTypeInfo<T> _typeInfo;

    /// <summary>Creates a serializer over a source-generated type info.</summary>
    /// <param name="typeInfo">The generated metadata for <typeparamref name="T"/>.</param>
    /// <exception cref="ArgumentNullException"><paramref name="typeInfo"/> is <see langword="null"/>.</exception>
    public SourceGenPayloadSerializer(JsonTypeInfo<T> typeInfo)
    {
        ArgumentNullException.ThrowIfNull(typeInfo);
        _typeInfo = typeInfo;
    }

    /// <inheritdoc />
    public byte[] Serialize(T value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return JsonSerializer.SerializeToUtf8Bytes(value, _typeInfo);
    }

    /// <inheritdoc />
    public T? Deserialize(ReadOnlySpan<byte> utf8Json) =>
        JsonSerializer.Deserialize(utf8Json, _typeInfo);
}
