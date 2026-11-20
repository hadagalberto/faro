namespace Faro.Payload;

/// <summary>
/// Converts payloads to and from their stored byte representation.
/// </summary>
/// <typeparam name="T">The payload type.</typeparam>
public interface IPayloadSerializer<T>
    where T : class
{
    /// <summary>Serializes <paramref name="value"/> into a new UTF-8 byte array.</summary>
    /// <param name="value">The payload to serialize.</param>
    /// <returns>The UTF-8 JSON representation.</returns>
    byte[] Serialize(T value);

    /// <summary>Deserializes a payload from UTF-8 JSON.</summary>
    /// <param name="utf8Json">The JSON bytes.</param>
    /// <returns>The deserialized payload, or <see langword="null"/> for JSON <c>null</c>.</returns>
    T? Deserialize(ReadOnlySpan<byte> utf8Json);
}
