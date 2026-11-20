using System.Runtime.InteropServices;
using Faro.Internal;

namespace Faro;

/// <summary>
/// Symmetric scalar (int8) quantizer with per-dimension ranges trained on a sample.
/// </summary>
/// <remarks>
/// <para>
/// Training measures, for every dimension, the smallest and largest value in the sample (or the
/// <c>quantile</c>-clipped bounds when a quantile is supplied) and derives a scale
/// <c>s[d] = max(|min[d]|, |max[d]|) / 127</c>. Encoding maps <c>x[d]</c> to
/// <c>clamp(round(x[d] / s[d]), -127, 127)</c> and stores one signed byte per dimension; a
/// constant-zero dimension uses a unit scale so no division by zero occurs. Decoding reconstructs
/// <c>code[d] * s[d]</c>.
/// </para>
/// <para>
/// Quantile clipping suppresses outliers: with <c>quantile = 0.995</c> the smallest and largest
/// 0.25% of a dimension's sample values are discarded before computing the range, so the single
/// byte is spent on the bulk of the distribution rather than on rare extremes.
/// </para>
/// <para>
/// Distances are computed with the int8 SIMD kernels in <see cref="Int8Kernels"/>: an integer dot
/// product for cosine/dot-product metrics and a squared-difference accumulation for Euclidean.
/// Cosine scores divide by both code norms so they are genuine approximate cosines in [0, 2].
/// </para>
/// </remarks>
public sealed class ScalarQuantizer : Quantizer
{
    private readonly float[] _scales;
    private readonly float[] _min;
    private readonly float[] _max;

    private ScalarQuantizer(int dimensions, Faro.Distance metric, float[] scales, float[] min, float[] max)
    {
        Dimensions = dimensions;
        Metric = metric;
        _scales = scales;
        _min = min;
        _max = max;
    }

    /// <inheritdoc />
    public override QuantizationKind Kind => QuantizationKind.Int8;

    /// <inheritdoc />
    public override int Dimensions { get; }

    /// <inheritdoc />
    public override Faro.Distance Metric { get; }

    /// <inheritdoc />
    public override int CodeLength => Dimensions;

    /// <summary>Per-dimension quantization scales (<c>max(|min|,|max|)/127</c>).</summary>
    public ReadOnlySpan<float> Scales => _scales;

    /// <summary>Per-dimension lower training bound (after quantile clipping).</summary>
    public ReadOnlySpan<float> Min => _min;

    /// <summary>Per-dimension upper training bound (after quantile clipping).</summary>
    public ReadOnlySpan<float> Max => _max;

    /// <summary>
    /// Trains a scalar quantizer on a sample of vectors.
    /// </summary>
    /// <param name="vectors">The training sample; every vector must have <paramref name="dimensions"/> finite values.</param>
    /// <param name="dimensions">Number of dimensions.</param>
    /// <param name="metric">Distance metric whose ordering convention codes must follow (not Hamming).</param>
    /// <param name="quantile">
    /// Central fraction of each dimension's sample used to compute the range (e.g. 0.995), or
    /// <see langword="null"/>/1 to use the raw min/max. Must be in [0.5, 1].
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="vectors"/> is null.</exception>
    /// <exception cref="ArgumentException">The sample is empty, a vector has the wrong length or a non-finite value, or the metric is Hamming.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An argument is outside its valid range.</exception>
    public static ScalarQuantizer Train(
        IEnumerable<float[]> vectors,
        int dimensions,
        Faro.Distance metric,
        double? quantile = null)
    {
        ArgumentNullException.ThrowIfNull(vectors);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);

        if (metric == Faro.Distance.Hamming)
        {
            throw new ArgumentException("Scalar int8 quantization does not support the Hamming metric; use BinaryQuantizer.", nameof(metric));
        }

        if (metric is not (Faro.Distance.Cosine or Faro.Distance.DotProduct or Faro.Distance.Euclidean))
        {
            throw new ArgumentException($"Unknown distance metric '{metric}'.", nameof(metric));
        }

        if (quantile.HasValue && (double.IsNaN(quantile.Value) || quantile.Value < 0.5 || quantile.Value > 1.0))
        {
            throw new ArgumentOutOfRangeException(nameof(quantile), quantile, "quantile must be in [0.5, 1].");
        }

        var sample = vectors.ToArray();
        if (sample.Length == 0)
        {
            throw new ArgumentException("The training sample must not be empty.", nameof(vectors));
        }

        foreach (float[] vector in sample)
        {
            if (vector is null)
            {
                throw new ArgumentException("The training sample must not contain null vectors.", nameof(vectors));
            }

            if (vector.Length != dimensions)
            {
                throw new ArgumentException($"every training vector must have {dimensions} dimensions, got {vector.Length}.", nameof(vectors));
            }

            for (int i = 0; i < vector.Length; i++)
            {
                if (!float.IsFinite(vector[i]))
                {
                    throw new ArgumentException($"training vector contains a non-finite value at index {i}.", nameof(vectors));
                }
            }
        }

        var scales = new float[dimensions];
        var min = new float[dimensions];
        var max = new float[dimensions];
        var column = new float[sample.Length];
        var sorted = new float[sample.Length];
        bool clip = quantile.HasValue && quantile.Value < 1.0;

        for (int d = 0; d < dimensions; d++)
        {
            for (int i = 0; i < sample.Length; i++)
            {
                column[i] = sample[i][d];
            }

            float low;
            float high;
            if (clip)
            {
                Array.Copy(column, sorted, column.Length);
                Array.Sort(sorted);
                double tail = (1.0 - quantile!.Value) * 0.5;
                int lowIndex = (int)Math.Floor(tail * (sorted.Length - 1));
                int highIndex = (int)Math.Ceiling((1.0 - tail) * (sorted.Length - 1));
                low = sorted[lowIndex];
                high = sorted[highIndex];
            }
            else
            {
                low = float.PositiveInfinity;
                high = float.NegativeInfinity;
                for (int i = 0; i < column.Length; i++)
                {
                    if (column[i] < low)
                    {
                        low = column[i];
                    }

                    if (column[i] > high)
                    {
                        high = column[i];
                    }
                }
            }

            min[d] = low;
            max[d] = high;
            float absMax = MathF.Max(MathF.Abs(low), MathF.Abs(high));
            scales[d] = absMax == 0f ? 1f : absMax / 127f;
        }

        return new ScalarQuantizer(dimensions, metric, scales, min, max);
    }

    /// <inheritdoc />
    public override void Encode(ReadOnlySpan<float> vector, Span<byte> destination)
    {
        ValidateVector(vector);
        ValidateDestination(destination);

        for (int i = 0; i < Dimensions; i++)
        {
            float scaled = vector[i] / _scales[i];
            float rounded = MathF.Round(scaled, MidpointRounding.AwayFromZero);
            if (rounded > 127f)
            {
                rounded = 127f;
            }
            else if (rounded < -127f)
            {
                rounded = -127f;
            }

            destination[i] = unchecked((byte)(sbyte)rounded);
        }
    }

    /// <inheritdoc />
    public override void EncodePacked(ReadOnlySpan<ulong> packed, Span<byte> destination)
    {
        throw new NotSupportedException("Scalar int8 quantization cannot encode packed sign bits.");
    }

    /// <inheritdoc />
    public override float Distance(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        ValidateCode(a);
        ValidateCode(b);

        var ca = MemoryMarshal.Cast<byte, sbyte>(a);
        var cb = MemoryMarshal.Cast<byte, sbyte>(b);
        int dot = Int8Kernels.Dot(ca, cb);

        return Metric switch
        {
            Faro.Distance.DotProduct => dot,
            Faro.Distance.Euclidean => Int8Kernels.SquaredDistance(ca, cb),
            Faro.Distance.Cosine => CosineFromCodes(ca, cb, dot),
            _ => throw new InvalidOperationException($"Scalar quantizer cannot score metric '{Metric}'."),
        };
    }

    /// <inheritdoc />
    public override void Decode(ReadOnlySpan<byte> code, Span<float> destination)
    {
        ValidateCode(code);
        ValidateDecodeDestination(destination);

        for (int i = 0; i < Dimensions; i++)
        {
            destination[i] = unchecked((sbyte)code[i]) * _scales[i];
        }
    }

    private static float CosineFromCodes(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b, int dot)
    {
        float normA = MathF.Sqrt(Int8Kernels.Dot(a, a));
        float normB = MathF.Sqrt(Int8Kernels.Dot(b, b));
        if (normA == 0f || normB == 0f)
        {
            return 1f;
        }

        float cos = dot / (normA * normB);
        if (cos > 1f)
        {
            cos = 1f;
        }
        else if (cos < -1f)
        {
            cos = -1f;
        }

        return 1f - cos;
    }
}
