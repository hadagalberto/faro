using System.Buffers.Binary;
using System.Text;

namespace Faro.Persistence;

/// <summary>Stable wire representation of a supported key type.</summary>
internal enum KeyKind : byte
{
    Int32 = 1,
    Int64 = 2,
    Guid = 3,
    String = 4,
}

/// <summary>
/// Binary codec for the key types supported by <see cref="FaroDb"/> collections.
/// </summary>
/// <remarks>
/// Keys are encoded deterministically: little-endian fixed-width integers, RFC-4122 byte order
/// for Guids, and length-prefixed UTF-8 for strings. The same encoding is used in payload pages and
/// WAL records, so replay never depends on a type-specific serializer.
/// </remarks>
internal static class KeyCodec<TKey>
{
    internal static readonly bool IsSupported =
        typeof(TKey) == typeof(int) ||
        typeof(TKey) == typeof(long) ||
        typeof(TKey) == typeof(Guid) ||
        typeof(TKey) == typeof(string);

    internal static KeyKind Kind =>
        typeof(TKey) == typeof(int) ? KeyKind.Int32 :
        typeof(TKey) == typeof(long) ? KeyKind.Int64 :
        typeof(TKey) == typeof(Guid) ? KeyKind.Guid :
        typeof(TKey) == typeof(string) ? KeyKind.String :
        throw Unsupported();

    internal static void Write(TKey key, ByteWriter writer)
    {
        switch (Kind)
        {
            case KeyKind.Int32:
                writer.WriteInt32((int)(object)key!);
                break;
            case KeyKind.Int64:
                writer.WriteInt64((long)(object)key!);
                break;
            case KeyKind.Guid:
                writer.WriteRaw(((Guid)(object)key!).ToByteArray());
                break;
            case KeyKind.String:
                writer.WriteString((string)(object)key!);
                break;
            default:
                throw Unsupported();
        }
    }

    internal static TKey Read(ref ByteReader reader)
    {
        return Kind switch
        {
            KeyKind.Int32 => (TKey)(object)reader.ReadInt32(),
            KeyKind.Int64 => (TKey)(object)reader.ReadInt64(),
            KeyKind.Guid => (TKey)(object)new Guid(reader.ReadRaw(16).Span),
            KeyKind.String => (TKey)(object)reader.ReadString(),
            _ => throw Unsupported(),
        };
    }

    internal static void Validate()
    {
        if (!IsSupported)
        {
            throw Unsupported();
        }
    }

    private static NotSupportedException Unsupported() =>
        new($"key type '{typeof(TKey)}' is not supported; use int, long, Guid or string");
}
