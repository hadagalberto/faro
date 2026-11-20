using System.Runtime.InteropServices;
using Faro.Internal;

namespace Faro;

/// <summary>
/// Binary (1-bit) sign quantizer: one bit per dimension, packed 64 bits per word.
/// </summary>
/// <remarks>
/// <para>
/// Encoding keeps only the sign of each component (via <see cref="BitVectors.PackSignBits(ReadOnlySpan{float}, Span{ulong})"/>);
/// decoding reconstructs ±1 per dimension. Code distances are popcount Hamming distances over
/// the packed words (<see cref="BitKernels"/>), which is the only operation needed for a first
/// pass over millions of candidates.
/// </para>
/// <para>
/// For normalized embeddings, sign agreement is strongly correlated with cosine similarity, so
/// the cheap Hamming pass followed by full-precision rescoring retains most of the recall while
/// using 1/32 of the memory of float32 vectors. DotProduct is rejected because Hamming is a
/// lower-is-better distance and cannot express that metric's ordering.
/// </para>
/// </remarks>
public sealed class BinaryQuantizer : Quantizer
{
    private readonly int _words;

    private BinaryQuantizer(int dimensions, Faro.Distance metric)
    {
        Dimensions = dimensions;
        Metric = metric;
        _words = BitVectors.WordsFor(dimensions);
    }

    /// <inheritdoc />
    public override QuantizationKind Kind => QuantizationKind.Binary;

    /// <inheritdoc />
    public override int Dimensions { get; }

    /// <inheritdoc />
    public override Faro.Distance Metric { get; }

    /// <inheritdoc />
    public override int CodeLength => _words * sizeof(ulong);

    /// <summary>Creates a binary quantizer for the given dimensions and metric.</summary>
    /// <param name="dimensions">Number of dimensions; must be positive.</param>
    /// <param name="metric">Cosine, Euclidean or Hamming (not DotProduct).</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimensions"/> is not positive.</exception>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is DotProduct or unknown.</exception>
    public static BinaryQuantizer Create(int dimensions, Faro.Distance metric)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);
        if (metric is Faro.Distance.DotProduct or not (Faro.Distance.Cosine or Faro.Distance.Euclidean or Faro.Distance.Hamming))
        {
            throw new ArgumentException($"Binary quantization does not support metric '{metric}'; use Cosine, Euclidean or Hamming.", nameof(metric));
        }

        return new BinaryQuantizer(dimensions, metric);
    }

    /// <inheritdoc />
    public override void Encode(ReadOnlySpan<float> vector, Span<byte> destination)
    {
        ValidateVector(vector);
        ValidateDestination(destination);

        BitVectors.PackSignBits(vector, MemoryMarshal.Cast<byte, ulong>(destination));
    }

    /// <inheritdoc />
    public override void EncodePacked(ReadOnlySpan<ulong> packed, Span<byte> destination)
    {
        ValidateDestination(destination);
        int words = BitVectors.WordsFor(Dimensions);
        if (packed.Length < words)
        {
            throw new ArgumentException($"packed must hold at least {words} words, got {packed.Length}.", nameof(packed));
        }

        packed[..words].CopyTo(MemoryMarshal.Cast<byte, ulong>(destination));
    }

    /// <inheritdoc />
    public override float Distance(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        ValidateCode(a);
        ValidateCode(b);

        return BitKernels.Hamming(
            MemoryMarshal.Cast<byte, ulong>(a),
            MemoryMarshal.Cast<byte, ulong>(b));
    }

    /// <inheritdoc />
    public override void Decode(ReadOnlySpan<byte> code, Span<float> destination)
    {
        ValidateCode(code);
        ValidateDecodeDestination(destination);

        BitVectors.UnpackSignBits(MemoryMarshal.Cast<byte, ulong>(code), Dimensions, destination);
    }
}
