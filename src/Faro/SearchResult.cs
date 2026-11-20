namespace Faro;

/// <summary>
/// One search hit returned by a vector index.
/// </summary>
/// <remarks>
/// Results are always ordered best-first: ascending <see cref="Score"/> for metrics where a
/// lower score is better (<see cref="Distance.Cosine"/>, <see cref="Distance.Euclidean"/>,
/// <see cref="Distance.Hamming"/>) and descending for <see cref="Distance.DotProduct"/>.
/// Exact ties are broken by ascending <see cref="Id"/> so results are deterministic.
/// </remarks>
public readonly record struct SearchResult
{
    /// <summary>Gets the internal identifier of the matched vector (the value returned by <c>Add</c>).</summary>
    public int Id { get; }

    /// <summary>
    /// Gets the metric value between the query and the matched vector: a distance for
    /// Cosine/Euclidean/Hamming, or a similarity for DotProduct.
    /// </summary>
    public float Score { get; }

    /// <summary>Creates a search hit.</summary>
    /// <param name="id">The internal vector identifier.</param>
    /// <param name="score">The metric score between the query and the vector.</param>
    public SearchResult(int id, float score)
    {
        Id = id;
        Score = score;
    }
}
