using System.Buffers;
using Faro.Internal;
using Xunit;

namespace Faro.Tests.Internal;

/// <summary>
/// Regression tests for the pool choice that makes steady-state search allocation-free.
/// </summary>
public class FaroPoolTests
{
    [Fact]
    public void Instance_IsPrivate_NotTheTrimmingSharedPool()
    {
        Assert.NotSame(ArrayPool<int>.Shared, FaroPool<int>.Instance);
        Assert.NotSame(ArrayPool<float>.Shared, FaroPool<float>.Instance);
        Assert.NotSame(ArrayPool<byte>.Shared, FaroPool<byte>.Instance);
        Assert.NotSame(ArrayPool<ulong>.Shared, FaroPool<ulong>.Instance);
        Assert.NotSame(ArrayPool<(float, int)>.Shared, FaroPool<(float, int)>.Instance);
    }

    [Fact]
    public void ReturnedBuffer_IsImmediatelyReused()
    {
        ArrayPool<int> pool = FaroPool<int>.Instance;
        int[] rented = pool.Rent(64);
        pool.Return(rented);
        int[] again = pool.Rent(64);
        try
        {
            Assert.Same(rented, again);
        }
        finally
        {
            pool.Return(again);
        }
    }
}
