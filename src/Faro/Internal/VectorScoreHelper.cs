namespace Faro.Internal;

/// <summary>
/// Scores a stored vector against a prepared query with the same kernels and ordering convention
/// as <see cref="HnswIndex"/> and <see cref="FlatIndex"/>. Used by the filtered exact planner path.
/// </summary>
internal static class VectorScoreHelper
{
    internal static float Score(
        Distance distance,
        bool normalize,
        ReadOnlySpan<float> preparedQuery,
        ReadOnlySpan<ulong> packedQuery,
        ReadOnlySpan<float> stored,
        Span<ulong> scratch)
    {
        switch (distance)
        {
            case Distance.Cosine:
                return normalize
                    ? 1f - VectorKernels.Dot(preparedQuery, stored)
                    : VectorKernels.Cosine(preparedQuery, stored);

            case Distance.DotProduct:
                return VectorKernels.Dot(preparedQuery, stored);

            case Distance.Euclidean:
                return VectorKernels.EuclideanSquared(preparedQuery, stored);

            case Distance.Hamming:
                BitVectors.PackSignBits(stored, scratch);
                return BitKernels.Hamming(packedQuery, scratch);

            default:
                throw new InvalidOperationException($"Unknown distance metric '{distance}'.");
        }
    }
}
