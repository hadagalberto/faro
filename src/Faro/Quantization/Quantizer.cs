namespace Faro;

/// <summary>
/// A trained codec that maps full-precision vectors to compact codes and computes approximate
/// distances directly on those codes.
/// </summary>
/// <remarks>
/// <para>
/// Quantizers are immutable once trained and safe to share across indexes and threads. A code is
/// a fixed-length <see langword="byte"/> buffer: int8 uses one byte per dimension (read back as
/// <see langword="sbyte"/>), binary uses packed sign bits (<see cref="BitVectors"/>).
/// </para>
/// <para>
/// <see cref="Distance(ReadOnlySpan{byte}, ReadOnlySpan{byte})"/> returns a score in the same
/// ordering convention as the collection's <see cref="Faro.Distance"/> metric — lower is closer
/// for cosine/Euclidean/Hamming, higher is closer for dot product — so indexes can rank
/// quantized candidates with their existing heaps. The score is approximate; searches rescore
/// the survivors against the original full-precision vectors.
/// </para>
/// </remarks>
public abstract class Quantizer
{
    /// <summary>The quantization scheme implemented by this codec.</summary>
    public abstract QuantizationKind Kind { get; }

    /// <summary>Number of dimensions the codec was trained/configured for.</summary>
    public abstract int Dimensions { get; }

    /// <summary>The metric whose ordering convention <see cref="Distance"/> follows.</summary>
    public abstract Faro.Distance Metric { get; }

    /// <summary>Number of bytes in one encoded code (equals <see cref="Dimensions"/> for int8).</summary>
    public abstract int CodeLength { get; }

    /// <summary>Encodes a full-precision vector into <paramref name="destination"/>.</summary>
    /// <param name="vector">The vector; must have <see cref="Dimensions"/> entries and finite values.</param>
    /// <param name="destination">Destination buffer; must have at least <see cref="CodeLength"/> bytes.</param>
    public abstract void Encode(ReadOnlySpan<float> vector, Span<byte> destination);

    /// <summary>
    /// Encodes a vector already packed as sign bits. Used by indexes whose stored form is packed
    /// (Hamming) and that cannot reconstruct the original floats on demand.
    /// </summary>
    /// <param name="packed">Packed sign bits covering <see cref="Dimensions"/> bits.</param>
    /// <param name="destination">Destination buffer; must have at least <see cref="CodeLength"/> bytes.</param>
    public abstract void EncodePacked(ReadOnlySpan<ulong> packed, Span<byte> destination);

    /// <summary>Returns the approximate metric score between two codes, in this codec's metric convention.</summary>
    /// <param name="a">First code; must have <see cref="CodeLength"/> bytes.</param>
    /// <param name="b">Second code; must have <see cref="CodeLength"/> bytes.</param>
    public abstract float Distance(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b);

    /// <summary>Reconstructs an approximate full-precision vector from a code.</summary>
    /// <param name="code">The code; must have <see cref="CodeLength"/> bytes.</param>
    /// <param name="destination">Destination span; must have at least <see cref="Dimensions"/> entries.</param>
    public abstract void Decode(ReadOnlySpan<byte> code, Span<float> destination);

    /// <summary>Validates the standard encode/decode preconditions shared by all codecs.</summary>
    protected void ValidateVector(ReadOnlySpan<float> vector)
    {
        if (vector.Length != Dimensions)
        {
            throw new ArgumentException($"vector must have {Dimensions} dimensions, got {vector.Length}.", nameof(vector));
        }

        for (int i = 0; i < vector.Length; i++)
        {
            if (!float.IsFinite(vector[i]))
            {
                throw new ArgumentException($"vector contains a non-finite value at index {i}.", nameof(vector));
            }
        }
    }

    /// <summary>Validates that a code buffer is large enough.</summary>
    protected void ValidateDestination(Span<byte> destination)
    {
        if (destination.Length < CodeLength)
        {
            throw new ArgumentException($"destination must hold at least {CodeLength} bytes, got {destination.Length}.", nameof(destination));
        }
    }

    /// <summary>Validates that a code has exactly the expected length.</summary>
    protected void ValidateCode(ReadOnlySpan<byte> code)
    {
        if (code.Length != CodeLength)
        {
            throw new ArgumentException($"code must have {CodeLength} bytes, got {code.Length}.", nameof(code));
        }
    }

    /// <summary>Validates that a decode destination is large enough.</summary>
    protected void ValidateDecodeDestination(Span<float> destination)
    {
        if (destination.Length < Dimensions)
        {
            throw new ArgumentException($"destination must hold at least {Dimensions} entries, got {destination.Length}.", nameof(destination));
        }
    }
}
