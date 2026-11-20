using Faro.Filtering;
using Faro.Hybrid;

namespace Faro;

/// <summary>
/// Optional tuning for a <see cref="FaroCollection{T}"/>.
/// </summary>
public sealed class CollectionOptions
{
    /// <summary>Custom selectivity planner. Defaults to <see cref="SearchPlanner"/> with its standard settings.</summary>
    public SearchPlanner? Planner { get; set; }

    /// <summary>Tokenizer for the full-text field. Defaults to <see cref="Tokenizer.Default"/>.</summary>
    public Tokenizer? Tokenizer { get; set; }

    /// <summary>Default search beam. When <see langword="null"/>, the index's own efSearch is used.</summary>
    public int? DefaultEfSearch { get; set; }

    /// <summary>Number of candidates each hybrid leg retrieves per requested hit. Defaults to 4.</summary>
    public int HybridCandidateMultiplier { get; set; } = 4;

    /// <summary>Minimum candidates each hybrid leg retrieves. Defaults to 50.</summary>
    public int MinHybridCandidates { get; set; } = 50;

    /// <summary>BM25 term-frequency saturation. Defaults to 1.5.</summary>
    public double? Bm25K1 { get; set; }

    /// <summary>BM25 document-length normalization. Defaults to 0.75.</summary>
    public double? Bm25B { get; set; }
}
