namespace Faro;

/// <summary>Construction options for <see cref="FlatIndex"/>.</summary>
public sealed class FlatOptions
{
    /// <summary>Number of dimensions of every vector. Required, must be positive.</summary>
    public required int Dimensions { get; set; }

    /// <summary>The distance metric used to compare vectors. Defaults to <see cref="Distance.Cosine"/>.</summary>
    public Distance Distance { get; set; } = Distance.Cosine;

    /// <summary>
    /// When <see langword="true"/>, vectors are normalized to unit length when added so cosine
    /// distance degenerates to <c>1 - dot</c>, the fastest exact path. When <see langword="null"/>
    /// (the default), it resolves to <see langword="true"/> for <see cref="Distance.Cosine"/> and
    /// <see langword="false"/> otherwise. Only meaningful for Cosine.
    /// </summary>
    /// <exception cref="ArgumentException">Set to <see langword="true"/> with a distance other than <see cref="Distance.Cosine"/>.</exception>
    public bool? NormalizeOnAdd { get; set; }

    /// <summary>Initial storage capacity, in vectors. Defaults to 1024. The index grows by doubling on demand.</summary>
    public int InitialCapacity { get; set; } = 1024;

    /// <summary>
    /// Searches on collections with at least this many vectors run in parallel across all CPU
    /// cores; below it they run single-threaded. Defaults to 4096. Set to 1 to force parallel
    /// search in tests or benchmarks.
    /// </summary>
    public int ParallelThreshold { get; set; } = 4096;

    /// <summary>
    /// Optional per-collection quantization. When set, searches score compact codes first and
    /// rescore the oversampled survivors against the stored full-precision vectors. A codec can
    /// be supplied pretrained (recommended for reproducibility) or trained lazily from the first
    /// vectors added to the index.
    /// </summary>
    public QuantizationOptions? Quantization { get; set; }
}
