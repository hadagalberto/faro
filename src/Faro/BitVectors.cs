namespace Faro;

/// <summary>
/// Helpers for packed bit vectors: 64 dimensions per <see langword="ulong"/> word, where bit
/// <c>i</c> of word <c>i/64</c> encodes the sign of dimension <c>i</c> (1 = sign bit set, 0 = not set).
/// </summary>
/// <remarks>
/// <para>
/// This is the storage form used for <see cref="Distance.Hamming"/> collections and later for
/// binary quantization: 1 bit per dimension, distances computed with popcount
/// (<see cref="Distances.Hamming"/>). The sign bit is taken from the raw IEEE 754 bits, so
/// negative zero and signed NaN set their bit too.
/// </para>
/// </remarks>
public static class BitVectors
{
    /// <summary>Returns the number of packed words needed for <paramref name="dimensions"/> bits.</summary>
    /// <param name="dimensions">Number of dimensions (bits).</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimensions"/> is negative.</exception>
    public static int WordsFor(int dimensions)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(dimensions);
        return (dimensions + 63) >> 6;
    }

    /// <summary>Packs the sign bits of <paramref name="vector"/> into a newly allocated word array.</summary>
    /// <param name="vector">The float vector whose signs to pack.</param>
    public static ulong[] PackSignBits(ReadOnlySpan<float> vector)
    {
        var packed = new ulong[WordsFor(vector.Length)];
        PackSignBits(vector, packed);
        return packed;
    }

    /// <summary>
    /// Packs the sign bits of <paramref name="vector"/> into <paramref name="destination"/>.
    /// Bits beyond <paramref name="vector"/>.Length in the last used word are zero.
    /// </summary>
    /// <param name="vector">The float vector whose signs to pack.</param>
    /// <param name="destination">The destination words; must hold at least <see cref="WordsFor"/>(length) entries. Extra entries are left untouched.</param>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is too small.</exception>
    public static void PackSignBits(ReadOnlySpan<float> vector, Span<ulong> destination)
    {
        int words = WordsFor(vector.Length);
        if (destination.Length < words)
        {
            throw new ArgumentException($"destination must hold at least {words} words, got {destination.Length}.", nameof(destination));
        }

        destination[..words].Clear();
        for (int i = 0; i < vector.Length; i++)
        {
            // The sign bit covers negatives, negative zero and signed NaN alike.
            if (BitConverter.SingleToInt32Bits(vector[i]) < 0)
            {
                destination[i >> 6] |= 1UL << (i & 63);
            }
        }
    }

    /// <summary>
    /// Reconstructs float dimensions from packed sign bits: <c>-1</c> where the bit is set,
    /// <c>+1</c> where it is not.
    /// </summary>
    /// <param name="packed">The packed words covering <paramref name="dimensions"/> bits.</param>
    /// <param name="dimensions">Number of dimensions (bits) to unpack.</param>
    /// <param name="destination">The destination span; must hold at least <paramref name="dimensions"/> entries.</param>
    /// <exception cref="ArgumentException">
    /// <paramref name="destination"/> is too small or <paramref name="packed"/> has fewer words than needed.
    /// </exception>
    public static void UnpackSignBits(ReadOnlySpan<ulong> packed, int dimensions, Span<float> destination)
    {
        if (destination.Length < dimensions)
        {
            throw new ArgumentException($"destination must hold at least {dimensions} entries, got {destination.Length}.", nameof(destination));
        }

        if (packed.Length < WordsFor(dimensions))
        {
            throw new ArgumentException($"packed must hold at least {WordsFor(dimensions)} words, got {packed.Length}.", nameof(packed));
        }

        for (int i = 0; i < dimensions; i++)
        {
            destination[i] = (packed[i >> 6] & (1UL << (i & 63))) != 0 ? -1f : 1f;
        }
    }
}
