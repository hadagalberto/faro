using System.Buffers.Binary;
using System.Text;

namespace Faro.Persistence;

/// <summary>Growable little-endian byte writer shared by the WAL and checkpoint encoder.</summary>
internal sealed class ByteWriter
{
    private byte[] _buffer;
    private int _length;

    internal ByteWriter(int capacity = 256)
    {
        _buffer = new byte[Math.Max(16, capacity)];
    }

    internal int Length => _length;

    internal ByteWriter WriteByte(byte value)
    {
        Ensure(1);
        _buffer[_length++] = value;
        return this;
    }

    internal ByteWriter WriteUInt16(ushort value)
    {
        Ensure(2);
        BinaryPrimitives.WriteUInt16LittleEndian(_buffer.AsSpan(_length), value);
        _length += 2;
        return this;
    }

    internal ByteWriter WriteInt32(int value)
    {
        Ensure(4);
        BinaryPrimitives.WriteInt32LittleEndian(_buffer.AsSpan(_length), value);
        _length += 4;
        return this;
    }

    internal ByteWriter WriteUInt32(uint value)
    {
        Ensure(4);
        BinaryPrimitives.WriteUInt32LittleEndian(_buffer.AsSpan(_length), value);
        _length += 4;
        return this;
    }

    internal ByteWriter WriteInt64(long value)
    {
        Ensure(8);
        BinaryPrimitives.WriteInt64LittleEndian(_buffer.AsSpan(_length), value);
        _length += 8;
        return this;
    }

    internal ByteWriter WriteUInt64(ulong value)
    {
        Ensure(8);
        BinaryPrimitives.WriteUInt64LittleEndian(_buffer.AsSpan(_length), value);
        _length += 8;
        return this;
    }

    internal ByteWriter WriteDouble(double value)
    {
        Ensure(8);
        BinaryPrimitives.WriteInt64LittleEndian(_buffer.AsSpan(_length), BitConverter.DoubleToInt64Bits(value));
        _length += 8;
        return this;
    }

    internal ByteWriter WriteBool(bool value) => WriteByte(value ? (byte)1 : (byte)0);

    internal ByteWriter WriteRaw(ReadOnlySpan<byte> bytes)
    {
        Ensure(bytes.Length);
        bytes.CopyTo(_buffer.AsSpan(_length));
        _length += bytes.Length;
        return this;
    }

    internal ByteWriter WriteBlob(ReadOnlySpan<byte> bytes) => WriteInt32(bytes.Length).WriteRaw(bytes);

    internal ByteWriter WriteString(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        int count = Encoding.UTF8.GetByteCount(value);
        WriteInt32(count);
        Ensure(count);
        Encoding.UTF8.GetBytes(value, _buffer.AsSpan(_length));
        _length += count;
        return this;
    }

    internal ByteWriter WriteNullableString(string? value) =>
        value is null ? WriteBool(false) : WriteBool(true).WriteString(value);

    internal ByteWriter WriteNullableDouble(double? value) =>
        value is null ? WriteBool(false) : WriteBool(true).WriteDouble(value.Value);

    internal ByteWriter WriteNullableLong(long? value) =>
        value is null ? WriteBool(false) : WriteBool(true).WriteInt64(value.Value);

    internal ReadOnlySpan<byte> Span => _buffer.AsSpan(0, _length);

    internal byte[] ToArray() => _buffer.AsSpan(0, _length).ToArray();

    private void Ensure(int count)
    {
        if ((long)_length + count <= _buffer.Length)
        {
            return;
        }

        long required = Math.Max((long)_buffer.Length * 2, (long)_length + count);
        int size = (int)Math.Min(required, Array.MaxLength);
        if (size < _length + count)
        {
            throw new OutOfMemoryException("persistence record exceeds the maximum array size");
        }

        Array.Resize(ref _buffer, size);
    }
}

/// <summary>Bounded little-endian byte reader; every read validates the remaining length.</summary>
internal sealed class ByteReader
{
    private readonly ReadOnlyMemory<byte> _memory;
    private int _offset;

    internal ByteReader(ReadOnlyMemory<byte> memory)
    {
        _memory = memory;
    }

    internal int Position => _offset;

    internal int Remaining => _memory.Length - _offset;

    internal bool EndOfStream => Remaining == 0;

    internal byte ReadByte()
    {
        Ensure(1);
        return _memory.Span[_offset++];
    }

    internal ushort ReadUInt16()
    {
        Ensure(2);
        ushort value = BinaryPrimitives.ReadUInt16LittleEndian(_memory.Span[_offset..]);
        _offset += 2;
        return value;
    }

    internal int ReadInt32()
    {
        Ensure(4);
        int value = BinaryPrimitives.ReadInt32LittleEndian(_memory.Span[_offset..]);
        _offset += 4;
        return value;
    }

    internal uint ReadUInt32()
    {
        Ensure(4);
        uint value = BinaryPrimitives.ReadUInt32LittleEndian(_memory.Span[_offset..]);
        _offset += 4;
        return value;
    }

    internal long ReadInt64()
    {
        Ensure(8);
        long value = BinaryPrimitives.ReadInt64LittleEndian(_memory.Span[_offset..]);
        _offset += 8;
        return value;
    }

    internal ulong ReadUInt64()
    {
        Ensure(8);
        ulong value = BinaryPrimitives.ReadUInt64LittleEndian(_memory.Span[_offset..]);
        _offset += 8;
        return value;
    }

    internal double ReadDouble()
    {
        Ensure(8);
        long bits = BinaryPrimitives.ReadInt64LittleEndian(_memory.Span[_offset..]);
        _offset += 8;
        return BitConverter.Int64BitsToDouble(bits);
    }

    internal bool ReadBool() => ReadByte() != 0;

    internal ReadOnlyMemory<byte> ReadRaw(int count)
    {
        Ensure(count);
        var slice = _memory.Slice(_offset, count);
        _offset += count;
        return slice;
    }

    internal ReadOnlyMemory<byte> ReadBlob()
    {
        int count = ReadInt32();
        if (count < 0)
        {
            throw new InvalidDataException("negative blob length");
        }

        return ReadRaw(count);
    }

    internal string ReadString()
    {
        ReadOnlyMemory<byte> bytes = ReadBlob();
        return Encoding.UTF8.GetString(bytes.Span);
    }

    internal string? ReadNullableString() => ReadBool() ? ReadString() : null;

    internal double? ReadNullableDouble() => ReadBool() ? ReadDouble() : null;

    internal long? ReadNullableLong() => ReadBool() ? ReadInt64() : null;

    private void Ensure(int count)
    {
        if (count < 0 || count > Remaining)
        {
            throw new InvalidDataException($"truncated persistence record: need {count} bytes, have {Remaining}.");
        }
    }
}
