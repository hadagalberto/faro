using System.Buffers;

namespace Faro.Internal;

/// <summary>
/// Private, non-trimming array pools for Faro's internal scratch buffers.
/// </summary>
/// <remarks>
/// <para>
/// The allocation-free guarantee of the span-based search overloads depends on every scratch
/// buffer coming from a pool that keeps it. <see cref="ArrayPool{T}.Shared"/> registers a gen2 GC
/// callback that periodically trims idle buffers and drops all of them under memory pressure. A
/// rent immediately after such a trim allocates a fresh array on the calling thread, which made
/// steady-state allocation assertions fail intermittently depending on unrelated GC activity.
/// </para>
/// <para>
/// Pools created by <see cref="ArrayPool{T}.Create()"/> never trim automatically, and keeping them
/// private to Faro means unrelated code cannot drain the same buckets. The number and size of the
/// buffers are bounded by the index configuration, so retaining them is intentional. All buffers
/// are returned in a <c>finally</c> block, including on the parallel and filtered search paths.
/// </para>
/// </remarks>
internal static class FaroPool<T>
{
    /// <summary>The private pool used by every Faro index for temporary search buffers.</summary>
    internal static readonly ArrayPool<T> Instance = ArrayPool<T>.Create();
}
