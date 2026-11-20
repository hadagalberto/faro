using Faro.Internal;

namespace Faro;

/// <summary>
/// Vector distance and similarity functions with SIMD acceleration and naive scalar reference
/// implementations.
/// </summary>
/// <remarks>
/// <para>
/// The SIMD path picks the widest hardware-accelerated vector (512/256/128-bit) and falls back
/// to a scalar loop otherwise. Each metric also has a <c>*Scalar</c> twin implemented with a
/// straightforward double-precision loop; the test suite asserts both paths agree within float
/// tolerance, and the scalar twins double as the exact-search ground truth for recall tests.
/// </para>
/// <para>
/// Inputs are not checked for <see cref="float.NaN"/> or infinities beyond the documented
/// argument validation; metrics propagate non-finite results naturally.
/// </para>
/// </remarks>
public static class Distances
{
    /// <summary>Returns the inner product (dot product) a·b — a similarity, larger means closer.</summary>
    /// <param name="a">Left vector.</param>
    /// <param name="b">Right vector.</param>
    /// <exception cref="ArgumentException">The vectors have different lengths.</exception>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        ValidateSameLength(a, b);
        return VectorKernels.Dot(a, b);
    }

    /// <summary>Returns the naive scalar inner product, accumulated in double precision (reference implementation).</summary>
    /// <param name="a">Left vector.</param>
    /// <param name="b">Right vector.</param>
    /// <exception cref="ArgumentException">The vectors have different lengths.</exception>
    public static float DotScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        ValidateSameLength(a, b);
        return VectorKernels.DotScalar(a, b);
    }

    /// <summary>
    /// Returns the cosine distance <c>1 - a·b/(|a||b|)</c>, in [0, 2]. Invariant to the magnitude
    /// of each vector. A zero-norm vector is treated as orthogonal to everything (distance 1).
    /// </summary>
    /// <param name="a">Left vector.</param>
    /// <param name="b">Right vector.</param>
    /// <exception cref="ArgumentException">The vectors have different lengths.</exception>
    public static float Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        ValidateSameLength(a, b);
        return VectorKernels.Cosine(a, b);
    }

    /// <summary>Returns the naive scalar cosine distance, computed in double precision (reference implementation).</summary>
    /// <param name="a">Left vector.</param>
    /// <param name="b">Right vector.</param>
    /// <exception cref="ArgumentException">The vectors have different lengths.</exception>
    public static float CosineScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        ValidateSameLength(a, b);

        double aa = 0;
        double bb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            aa += (double)a[i] * a[i];
            bb += (double)b[i] * b[i];
        }

        if (aa == 0 || bb == 0)
        {
            return 1f;
        }

        double dot = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
        }

        double cos = dot / Math.Sqrt(aa * bb);
        if (cos > 1)
        {
            cos = 1;
        }
        else if (cos < -1)
        {
            cos = -1;
        }

        return (float)(1 - cos);
    }

    /// <summary>
    /// Returns the squared Euclidean (L2) distance: the sum of squared component differences.
    /// Monotonic in the true L2 distance, so rankings are identical and the square root is saved.
    /// </summary>
    /// <param name="a">Left vector.</param>
    /// <param name="b">Right vector.</param>
    /// <exception cref="ArgumentException">The vectors have different lengths.</exception>
    public static float EuclideanSquared(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        ValidateSameLength(a, b);
        return VectorKernels.EuclideanSquared(a, b);
    }

    /// <summary>Returns the naive scalar squared L2 distance, accumulated in double precision (reference implementation).</summary>
    /// <param name="a">Left vector.</param>
    /// <param name="b">Right vector.</param>
    /// <exception cref="ArgumentException">The vectors have different lengths.</exception>
    public static float EuclideanSquaredScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        ValidateSameLength(a, b);
        return VectorKernels.EuclideanSquaredScalar(a, b);
    }

    /// <summary>
    /// Returns the Hamming distance between two packed bit vectors (see <see cref="BitVectors"/>) —
    /// the number of differing bits, one bit per dimension.
    /// </summary>
    /// <param name="a">Left packed vector.</param>
    /// <param name="b">Right packed vector.</param>
    /// <exception cref="ArgumentException">The packed vectors have different lengths (different dimensions).</exception>
    public static int Hamming(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        ValidateSameLength(a, b);
        return BitKernels.Hamming(a, b);
    }

    /// <summary>Returns the naive scalar Hamming distance (reference implementation).</summary>
    /// <param name="a">Left packed vector.</param>
    /// <param name="b">Right packed vector.</param>
    /// <exception cref="ArgumentException">The packed vectors have different lengths (different dimensions).</exception>
    public static int HammingScalar(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        ValidateSameLength(a, b);
        return BitKernels.HammingScalar(a, b);
    }

    /// <summary>Returns the L2 norm (Euclidean length) of a vector.</summary>
    /// <param name="vector">The vector.</param>
    public static float Norm(ReadOnlySpan<float> vector) => MathF.Sqrt(VectorKernels.SumSquares(vector));

    /// <summary>
    /// Normalizes a vector to unit length, in place. A zero vector is left unchanged (its norm
    /// cannot be scaled away) and <c>0</c> is returned for it.
    /// </summary>
    /// <param name="vector">The vector to normalize in place.</param>
    /// <returns>The norm of the vector before normalization.</returns>
    public static float Normalize(Span<float> vector)
    {
        float sumSquares = VectorKernels.SumSquares(vector);
        if (sumSquares == 0f)
        {
            return 0f;
        }

        float norm = MathF.Sqrt(sumSquares);
        float inverse = 1f / norm;
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= inverse;
        }

        return norm;
    }

    /// <summary>
    /// Copies <paramref name="source"/> into <paramref name="destination"/> scaled to unit length.
    /// A zero source vector is copied unchanged.
    /// </summary>
    /// <param name="source">The vector to read.</param>
    /// <param name="destination">The destination span; must have the same length as <paramref name="source"/>.</param>
    /// <exception cref="ArgumentException">The spans have different lengths.</exception>
    public static void CopyNormalized(ReadOnlySpan<float> source, Span<float> destination)
    {
        if (source.Length != destination.Length)
        {
            throw new ArgumentException($"destination must have the same length as source ({source.Length}), got {destination.Length}.", nameof(destination));
        }

        float sumSquares = VectorKernels.SumSquares(source);
        if (sumSquares == 0f)
        {
            source.CopyTo(destination);
            return;
        }

        float inverse = 1f / MathF.Sqrt(sumSquares);
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = source[i] * inverse;
        }
    }

    /// <summary>Returns whether a lower metric value means "closer" for the given distance kind (false for DotProduct).</summary>
    /// <param name="distance">The distance kind.</param>
    public static bool LowerIsCloser(Distance distance) => distance != Distance.DotProduct;

    private static void ValidateSameLength(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"vectors must have the same length, got {a.Length} and {b.Length}.");
        }
    }

    private static void ValidateSameLength(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"packed bit vectors must have the same length, got {a.Length} and {b.Length}.");
        }
    }
}
