namespace Faro;

/// <summary>One search hit addressed by the caller's key instead of an internal vector id.</summary>
/// <typeparam name="TKey">The collection's key type.</typeparam>
public readonly record struct SearchHit<TKey>
{
    /// <summary>The key of the matching record.</summary>
    public TKey Key { get; init; }

    /// <summary>The metric score; lower is better for distances, higher for DotProduct.</summary>
    public float Score { get; init; }
}

/// <summary>One hybrid hit addressed by the caller's key, carrying both ranking signals.</summary>
/// <typeparam name="TKey">The collection's key type.</typeparam>
public readonly record struct HybridHit<TKey>
{
    /// <summary>The key of the matching record.</summary>
    public TKey Key { get; init; }

    /// <summary>The fused score used for ordering (higher is better).</summary>
    public float Score { get; init; }

    /// <summary>The raw vector metric score, when the record appears in the vector ranking.</summary>
    public float? VectorScore { get; init; }

    /// <summary>The raw BM25 score, when the record appears in the text ranking.</summary>
    public double? TextScore { get; init; }

    /// <summary>One-based rank in the vector list, or zero when absent.</summary>
    public int VectorRank { get; init; }

    /// <summary>One-based rank in the BM25 list, or zero when absent.</summary>
    public int TextRank { get; init; }
}
