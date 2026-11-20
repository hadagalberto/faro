namespace Faro.Persistence;

/// <summary>
/// Immutable descriptor of a node's stored vector. The data is either an owned array (in-memory
/// mode, WAL replays and newly inserted nodes) or a slice of a <see cref="MappedVectorRegion"/>
/// belonging to the current checkpoint.
/// </summary>
internal sealed class VectorData
{
    private static readonly VectorData EmptyInstance = new([], null, null, 0, 0, isBits: false);

    private readonly float[]? _floats;
    private readonly ulong[]? _bits;
    private readonly MappedVectorRegion? _region;
    private readonly long _offsetBytes;
    private readonly int _count;
    private readonly bool _isBits;

    private VectorData(float[]? floats, ulong[]? bits, MappedVectorRegion? region, long offsetBytes, int count, bool isBits)
    {
        _floats = floats;
        _bits = bits;
        _region = region;
        _offsetBytes = offsetBytes;
        _count = count;
        _isBits = isBits;
    }

    internal static VectorData Empty => EmptyInstance;

    internal bool IsMapped => _region is not null;

    internal bool IsBits => _isBits;

    internal long MappedByteOffset => _offsetBytes;

    internal int Count => _count;

    internal static VectorData FromFloats(float[] floats) => new(floats, null, null, 0, floats.Length, false);

    internal static VectorData FromBits(ulong[] bits) => new(null, bits, null, 0, bits.Length, true);

    internal static VectorData MappedFloats(MappedVectorRegion region, long byteOffset, int count) =>
        new(null, null, region, byteOffset, count, false);

    internal static VectorData MappedBits(MappedVectorRegion region, long byteOffset, int count) =>
        new(null, null, region, byteOffset, count, true);

    internal VectorData Remap(MappedVectorRegion region) => _isBits
        ? MappedBits(region, _offsetBytes, _count)
        : MappedFloats(region, _offsetBytes, _count);

    internal ReadOnlySpan<float> Floats
    {
        get
        {
            if (_isBits)
            {
                throw new InvalidOperationException("this vector stores packed bits");
            }

            return _region is null
                ? _floats!.AsSpan(0, _count)
                : _region.Floats(_offsetBytes, _count);
        }
    }

    internal ReadOnlySpan<ulong> Bits
    {
        get
        {
            if (!_isBits)
            {
                throw new InvalidOperationException("this vector stores float32 values");
            }

            return _region is null
                ? _bits!.AsSpan(0, _count)
                : _region.Ulongs(_offsetBytes, _count);
        }
    }
}
