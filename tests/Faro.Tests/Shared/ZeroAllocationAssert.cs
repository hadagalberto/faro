using System.Runtime.CompilerServices;
using Xunit;

namespace Faro.Testing;

/// <summary>
/// Shared harness for every steady-state allocation assertion.
/// </summary>
/// <remarks>
/// <para>
/// Warm-up runs with the exact delegate used for measurement, so tiered-JIT/OSR transitions,
/// lazy static initialization and first-use pool growth all complete before the counter starts.
/// A forced gen2 collection sits between warm-up and measurement: the shared array pool trims on
/// gen2, which used to drop the cached buffers and make the next rent allocate. Faro uses private
/// non-trimming pools (see <c>FaroPool&lt;T&gt;</c>), so a collection must not change the result.
/// </para>
/// <para>
/// A large number of measured iterations makes a single stray allocation visible and averages
/// out counter noise. The assertion reports the exact byte count instead of a bare boolean.
/// </para>
/// </remarks>
public static class ZeroAllocationAssert
{
    private const int DefaultWarmupIterations = 10_000;
    private const int DefaultMeasuredIterations = 10_000;

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Run(
        Action operation,
        int warmupIterations = DefaultWarmupIterations,
        int measuredIterations = DefaultMeasuredIterations)
    {
        ArgumentNullException.ThrowIfNull(operation);
        Assert.True(warmupIterations > 0, "warmupIterations must be positive.");
        Assert.True(measuredIterations > 0, "measuredIterations must be positive.");

        for (int i = 0; i < warmupIterations; i++)
        {
            operation();
        }

        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: false);
        GC.WaitForPendingFinalizers();

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < measuredIterations; i++)
        {
            operation();
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(
            allocated == 0,
            $"Steady state allocated {allocated} bytes over {measuredIterations} iterations " +
            $"({(allocated == 0 ? 0 : allocated / measuredIterations)} bytes/iteration).");
    }
}
