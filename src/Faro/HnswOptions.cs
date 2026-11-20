namespace Faro;

/// <summary>Construction options for <see cref="HnswIndex"/>.</summary>
/// <remarks>
/// Defaults follow Malkov &amp; Yashunin (2016/2018): <c>M = 16</c>,
/// <c>efConstruction = 200</c>, level multiplier <c>1 / ln(M)</c>. Neighbour selection uses
/// Algorithm 4 (the heuristic that keeps candidates closer to the query than to any already
/// chosen neighbour) with <see cref="ExtendCandidates"/> enabled, matching hnswlib's defaults.
/// </remarks>
public sealed class HnswOptions
{
    /// <summary>Number of dimensions of every vector. Required, must be positive.</summary>
    public required int Dimensions { get; set; }

    /// <summary>The distance metric used to compare vectors. Defaults to <see cref="Distance.Cosine"/>.</summary>
    public Distance Distance { get; set; } = Distance.Cosine;

    /// <summary>
    /// Normalize vectors on insertion so cosine distance degenerates to <c>1 - dot</c>. When
    /// <see langword="null"/> (the default) it resolves to <see langword="true"/> for cosine and
    /// <see langword="false"/> otherwise. Only meaningful for <see cref="Distance.Cosine"/>.
    /// </summary>
    public bool? NormalizeOnAdd { get; set; }

    /// <summary>
    /// Maximum number of neighbours per node on layers above 0 (<c>M</c>). Layer 0 allows
    /// <c>M0 = 2 * M</c>. Must be at least 2. Defaults to 16.
    /// </summary>
    public int M { get; set; } = 16;

    /// <summary>Size of the candidate beam during construction (Algorithm 5). Must be positive. Defaults to 200.</summary>
    public int EfConstruction { get; set; } = 200;

    /// <summary>Search beam width. Must be positive. Defaults to 64.</summary>
    public int EfSearch { get; set; } = 64;

    /// <summary>
    /// Level generation multiplier <c>mL</c>. When <see langword="null"/> (the default) it is
    /// <c>1 / ln(M)</c>, the value recommended by the paper. Must be positive when supplied.
    /// </summary>
    public double? LevelMultiplier { get; set; }

    /// <summary>
    /// Fixed seed for level assignment and tie-breaking. When supplied, inserting the same
    /// vectors in the same order produces a bit-identical graph; when <see langword="null"/>
    /// levels come from a shared process-wide random source.
    /// </summary>
    public long? Seed { get; set; }

    /// <summary>
    /// Algorithm 4's <c>extendCandidates</c>: also consider the neighbours of candidate nodes
    /// when choosing the new node's links. Improves recall at a modest build cost.
    /// Defaults to <see langword="true"/>.
    /// </summary>
    public bool ExtendCandidates { get; set; } = true;

    /// <summary>
    /// Algorithm 4's <c>keepPrunedConnections</c>: fill leftover slots with the best candidates
    /// that failed the diversity test. Defaults to <see langword="false"/>.
    /// </summary>
    public bool KeepPrunedConnections { get; set; }

    /// <summary>
    /// Algorithm 4's diversity factor <c>mc</c>. A candidate is admitted when its distance to
    /// the query is smaller than <c>mc</c> times its distance to every already-chosen neighbour.
    /// Must be at least 1. Defaults to 1.
    /// </summary>
    public double Mc { get; set; } = 1.0;

    /// <summary>Initial node capacity. Defaults to 1024. Storage grows by doubling on demand.</summary>
    public int InitialCapacity { get; set; } = 1024;

    /// <summary>
    /// Optional per-collection quantization. Graph construction always uses full precision;
    /// searches route with compact codes and rescore an oversampled candidate set against the
    /// stored vectors. A codec can be supplied pretrained or trained lazily from the first nodes
    /// added to the index.
    /// </summary>
    public QuantizationOptions? Quantization { get; set; }
}
