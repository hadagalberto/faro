namespace Faro.Persistence;

/// <summary>One stored payload: internal vector id, encoded user key and serialized payload bytes.</summary>
internal sealed class PersistedRecord
{
    internal PersistedRecord(int internalId, byte[] key, byte[] payload)
    {
        InternalId = internalId;
        Key = key;
        Payload = payload;
    }

    internal int InternalId { get; }
    internal byte[] Key { get; }
    internal byte[] Payload { get; }

    /// <summary>Stable hex form used for key lookups independent of the key's CLR type.</summary>
    internal string KeyToken => Convert.ToHexString(Key);

    internal void Write(ByteWriter writer)
    {
        writer.WriteInt32(InternalId);
        writer.WriteBlob(Key);
        writer.WriteBlob(Payload);
    }

    internal static PersistedRecord Read(ByteReader reader) =>
        new(reader.ReadInt32(), reader.ReadBlob().ToArray(), reader.ReadBlob().ToArray());
}
