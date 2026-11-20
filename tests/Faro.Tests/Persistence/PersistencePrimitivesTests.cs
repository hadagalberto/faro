using Faro.Persistence;
using Xunit;

namespace Faro.Tests.Persistence;

public class PersistencePrimitivesTests
{
    [Fact]
    public void Crc32C_MatchesCastagnoliTestVector()
    {
        // CRC32C("123456789") = 0xE3069283 (Castagnoli et al., 1993).
        Assert.Equal(0xE3069283u, Crc32C.Compute("123456789"u8));
        Assert.Equal(0u, Crc32C.Compute(ReadOnlySpan<byte>.Empty));
        Assert.Equal(Crc32C.Compute("abcdef"u8, "ghij"u8), Crc32C.Compute("abcdefghij"u8));
    }

    [Fact]
    public void ByteWriterReader_RoundTripAllPrimitives()
    {
        var writer = new ByteWriter();
        writer.WriteByte(0xAB).WriteUInt16(0xCDEF).WriteInt32(-12345).WriteUInt32(0xDEADBEEF)
              .WriteInt64(long.MinValue + 7).WriteUInt64(ulong.MaxValue - 3)
              .WriteDouble(Math.PI).WriteBool(true).WriteString("faro\u00e7\u00e3o")
              .WriteBlob([1, 2, 3, 4]).WriteNullableString(null).WriteNullableDouble(null)
              .WriteNullableLong(42);

        var reader = new ByteReader(writer.ToArray());
        Assert.Equal(0xAB, reader.ReadByte());
        Assert.Equal(0xCDEF, reader.ReadUInt16());
        Assert.Equal(-12345, reader.ReadInt32());
        Assert.Equal(0xDEADBEEFu, reader.ReadUInt32());
        Assert.Equal(long.MinValue + 7, reader.ReadInt64());
        Assert.Equal(ulong.MaxValue - 3, reader.ReadUInt64());
        Assert.Equal(Math.PI, reader.ReadDouble());
        Assert.True(reader.ReadBool());
        Assert.Equal("faro\u00e7\u00e3o", reader.ReadString());
        Assert.Equal(new byte[] { 1, 2, 3, 4 }, reader.ReadBlob().ToArray());
        Assert.Null(reader.ReadNullableString());
        Assert.Null(reader.ReadNullableDouble());
        Assert.Equal(42, reader.ReadNullableLong());
        Assert.True(reader.EndOfStream);
        Assert.Throws<InvalidDataException>(() => reader.ReadInt32());
    }

    [Theory]
    [InlineData(7)]
    [InlineData(-42)]
    public void KeyCodec_Int32RoundTrips(int key)
    {
        var writer = new ByteWriter();
        KeyCodec<int>.Write(key, writer);
        var reader = new ByteReader(writer.ToArray());
        Assert.Equal(key, KeyCodec<int>.Read(ref reader));
        Assert.True(reader.EndOfStream);
    }

    [Fact]
    public void KeyCodec_AllSupportedTypesRoundTrip()
    {
        Assert.Equal(long.MinValue, RoundTrip<long>(long.MinValue));
        Assert.Equal("tenant/\u00e1", RoundTrip<string>("tenant/\u00e1"));
        Guid guid = Guid.Parse("11f0c2d4-5a6b-4c7d-8e9f-0a1b2c3d4e5f");
        Assert.Equal(guid, RoundTrip<Guid>(guid));
        Assert.False(KeyCodec<double>.IsSupported);
        Assert.Throws<NotSupportedException>(KeyCodec<double>.Validate);
    }

    private static TKey RoundTrip<TKey>(TKey key)
        where TKey : notnull
    {
        var writer = new ByteWriter();
        KeyCodec<TKey>.Write(key, writer);
        var reader = new ByteReader(writer.ToArray());
        TKey result = KeyCodec<TKey>.Read(ref reader);
        Assert.True(reader.EndOfStream);
        return result;
    }
}
