namespace Faro;

/// <summary>
/// Per-collection quantization configuration: the codec kind, its training quantile (int8 only),
/// the oversampling factor used before full-precision rescoring, and an optional pretrained codec.
/// </summary>
public abstract class QuantizationOptions
{
    /// <summary>Creates quantization options.</summary>
    /// <param name="rescore">Oversampling factor; at least k × this many candidates are rescored.</param>
    /// <param name="quantile">Central training fraction for int8, or <see langword="null"/> for raw min/max.</param>
    /// <param name="preTrained">An already trained codec, when the caller trained one explicitly.</param>
    protected QuantizationOptions(int rescore, double? quantile, Quantizer? preTrained)
    {
        if (rescore < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(rescore), rescore, "Rescore factor must be at least 1.");
        }

        if (quantile.HasValue && (double.IsNaN(quantile.Value) || quantile.Value < 0.5 || quantile.Value > 1.0))
        {
            throw new ArgumentOutOfRangeException(nameof(quantile), quantile, "quantile must be in [0.5, 1].");
        }

        RescoreFactor = rescore;
        Quantile = quantile;
        PreTrained = preTrained;
    }

    /// <summary>The quantization scheme to use.</summary>
    public abstract QuantizationKind Kind { get; }

    /// <summary>
    /// Oversampling factor: the first pass keeps <c>k * RescoreFactor</c> quantized candidates and
    /// the second pass rescores them with full precision before selecting the final <c>k</c>.
    /// </summary>
    public int RescoreFactor { get; }

    /// <summary>Central fraction of each dimension's sample used to compute int8 ranges.</summary>
    public double? Quantile { get; }

    /// <summary>The pretrained codec supplied by the caller, if any.</summary>
    public Quantizer? PreTrained { get; }

    /// <summary>Validates that a pretrained codec (when present) matches the collection.</summary>
    internal void ValidateCompatibility(int dimensions, Faro.Distance metric)
    {
        if (PreTrained is null)
        {
            return;
        }

        if (PreTrained.Kind != Kind)
        {
            throw new ArgumentException($"quantization kind is {Kind} but the pretrained codec is {PreTrained.Kind}.", nameof(PreTrained));
        }

        if (PreTrained.Dimensions != dimensions)
        {
            throw new ArgumentException($"pretrained codec has {PreTrained.Dimensions} dimensions, expected {dimensions}.", nameof(PreTrained));
        }

        if (PreTrained.Metric != metric)
        {
            throw new ArgumentException($"pretrained codec uses metric '{PreTrained.Metric}', expected '{metric}'.", nameof(PreTrained));
        }
    }

    /// <summary>Returns the code length in bytes for a collection of the given shape.</summary>
    internal abstract int CodeLengthFor(int dimensions, Faro.Distance metric);

    /// <summary>Trains (or returns) the codec for a collection of the given shape.</summary>
    internal abstract Quantizer Train(IReadOnlyList<float[]> sample, int dimensions, Faro.Distance metric);
}

internal sealed class Int8QuantizationOptions : QuantizationOptions
{
    internal Int8QuantizationOptions(int rescore, double? quantile, Quantizer? preTrained)
        : base(rescore, quantile, preTrained)
    {
    }

    public override QuantizationKind Kind => QuantizationKind.Int8;

    internal override int CodeLengthFor(int dimensions, Faro.Distance metric)
    {
        if (metric == Faro.Distance.Hamming)
        {
            throw new ArgumentException("Scalar int8 quantization does not support the Hamming metric; use binary quantization.", "options");
        }

        return dimensions;
    }

    internal override Quantizer Train(IReadOnlyList<float[]> sample, int dimensions, Faro.Distance metric) =>
        PreTrained ?? ScalarQuantizer.Train(sample, dimensions, metric, Quantile);
}

internal sealed class BinaryQuantizationOptions : QuantizationOptions
{
    internal BinaryQuantizationOptions(int rescore, Quantizer? preTrained)
        : base(rescore, null, preTrained)
    {
    }

    public override QuantizationKind Kind => QuantizationKind.Binary;

    internal override int CodeLengthFor(int dimensions, Faro.Distance metric) =>
        BinaryQuantizer.Create(dimensions, metric).CodeLength;

    internal override Quantizer Train(IReadOnlyList<float[]> sample, int dimensions, Faro.Distance metric) =>
        PreTrained ?? BinaryQuantizer.Create(dimensions, metric);
}

/// <summary>Factory for per-collection quantization options.</summary>
/// <remarks>
/// When no pretrained codec is supplied, indexes train lazily on a bounded sample of the vectors
/// added before the first search, then encode the whole collection once.
/// </remarks>
public static class Quantization
{
    /// <summary>Scalar int8 quantization with oversampling and full-precision rescoring.</summary>
    /// <param name="rescore">Oversampling factor (default 4).</param>
    /// <param name="quantile">Central training fraction, e.g. 0.995; <see langword="null"/> uses raw min/max.</param>
    public static QuantizationOptions Int8(int rescore = 4, double? quantile = null) =>
        new Int8QuantizationOptions(rescore, quantile, null);

    /// <summary>Scalar int8 quantization using a codec trained explicitly by the caller.</summary>
    /// <param name="trained">The trained scalar quantizer.</param>
    /// <param name="rescore">Oversampling factor (default 4).</param>
    public static QuantizationOptions Int8(ScalarQuantizer trained, int rescore = 4)
    {
        ArgumentNullException.ThrowIfNull(trained);
        return new Int8QuantizationOptions(rescore, null, trained);
    }

    /// <summary>Binary sign quantization with popcount Hamming distances and rescoring.</summary>
    /// <param name="rescore">Oversampling factor (default 8).</param>
    public static QuantizationOptions Binary(int rescore = 8) =>
        new BinaryQuantizationOptions(rescore, null);

    /// <summary>Binary sign quantization using a codec created explicitly by the caller.</summary>
    /// <param name="trained">The binary quantizer.</param>
    /// <param name="rescore">Oversampling factor (default 8).</param>
    public static QuantizationOptions Binary(BinaryQuantizer trained, int rescore = 8)
    {
        ArgumentNullException.ThrowIfNull(trained);
        return new BinaryQuantizationOptions(rescore, trained);
    }
}
