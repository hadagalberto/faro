namespace Faro.Hybrid;

/// <summary>
/// One hit returned by hybrid search, carrying both ranking signals.
/// </summary>
public readonly record struct HybridResult
{
    /// <summary>The vector id shared by both indexes.</summary>
    public int Id { get; init; }

    /// <summary>The fused score used for ordering (higher is better).</summary>
    public float Score { get; init; }

    /// <summary>The raw vector metric score, or <see langword="null"/> when absent from the vector ranking.</summary>
    public float? VectorScore { get; init; }

    /// <summary>The raw BM25 score, or <see langword="null"/> when absent from the text ranking.</summary>
    public double? TextScore { get; init; }

    /// <summary>One-based rank in the vector list, or zero when absent.</summary>
    public int VectorRank { get; init; }

    /// <summary>One-based rank in the BM25 list, or zero when absent.</summary>
    public int TextRank { get; init; }
}
