namespace Faro;

/// <summary>Shared one-time training logic used by quantized indexes.</summary>
internal static class QuantizationRuntime
{
    /// <summary>
    /// Maximum number of vectors retained for lazy training. The bound keeps memory predictable
    /// on large collections while still giving int8 per-dimension ranges a reliable sample.
    /// </summary>
    internal const int TrainingSampleSize = 4096;

    internal static Quantizer Train(
        QuantizationOptions options,
        IReadOnlyList<float[]> sample,
        int dimensions,
        Faro.Distance metric)
    {
        if (options.PreTrained is not null)
        {
            // Pretrained codecs are validated by the index constructor and carry their own data.
            return options.PreTrained;
        }

        if (sample.Count == 0)
        {
            throw new InvalidOperationException(
                "Cannot train a quantizer without vectors: add at least one vector before searching, or supply a pretrained codec.");
        }

        return options.Train(sample, dimensions, metric);
    }
}
