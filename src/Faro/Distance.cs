namespace Faro;

/// <summary>
/// The distance/similarity metric used to compare vectors.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Cosine"/>, <see cref="Euclidean"/> and <see cref="Hamming"/> are distances
/// (lower value means closer). <see cref="DotProduct"/> is a similarity (higher value means
/// closer); use <see cref="Faro.Distances.LowerIsCloser"/> to query the ordering convention of
/// a metric.
/// </para>
/// <para>
/// <see cref="Euclidean"/> is the <em>squared</em> L2 distance: the square root is omitted
/// because it is monotonic and preserves ranking while being considerably cheaper to compute.
/// </para>
/// </remarks>
public enum Distance
{
    /// <summary>One minus the cosine of the angle between the two vectors, in [0, 2]. Invariant to vector magnitude.</summary>
    Cosine,

    /// <summary>Inner product. A similarity: larger means closer. Best for pre-normalized embeddings.</summary>
    DotProduct,

    /// <summary>Squared Euclidean (L2) distance: the sum of squared component differences. A distance: smaller means closer.</summary>
    Euclidean,

    /// <summary>Hamming distance over the sign bits of the vectors (1 bit per dimension, packed 64 dimensions per word). A distance: smaller means closer.</summary>
    Hamming,
}
