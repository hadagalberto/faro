using Faro.Internal;
using Xunit;

namespace Faro.Tests.DistanceTests;

/// <summary>
/// SIMD kernels only had Debug.Assert for equal lengths, then used LoadUnsafe. In Release the
/// assert disappears and a short right-hand span is read out of bounds. Public Distances methods
/// validate, but the kernels are independently callable, so they must validate too.
/// </summary>
public class KernelLengthValidationTests
{
    [Fact]
    public void FloatKernels_RejectMismatchedLengthsInRelease()
    {
        var longVector = new float[128];
        var shortVector = new float[8];

        Assert.Throws<ArgumentException>(() => VectorKernels.DotV128(longVector, shortVector));
        Assert.Throws<ArgumentException>(() => VectorKernels.DotV256(longVector, shortVector));
        Assert.Throws<ArgumentException>(() => VectorKernels.DotV512(longVector, shortVector));
        Assert.Throws<ArgumentException>(() => VectorKernels.EuclideanSquaredV128(longVector, shortVector));
        Assert.Throws<ArgumentException>(() => VectorKernels.EuclideanSquaredV256(longVector, shortVector));
        Assert.Throws<ArgumentException>(() => VectorKernels.EuclideanSquaredV512(longVector, shortVector));
    }

    [Fact]
    public void Int8Kernels_RejectMismatchedLengthsInRelease()
    {
        var longCodes = new sbyte[128];
        var shortCodes = new sbyte[8];

        Assert.Throws<ArgumentException>(() => Int8Kernels.DotV128(longCodes, shortCodes));
        Assert.Throws<ArgumentException>(() => Int8Kernels.DotV256(longCodes, shortCodes));
        Assert.Throws<ArgumentException>(() => Int8Kernels.DotV512(longCodes, shortCodes));
        Assert.Throws<ArgumentException>(() => Int8Kernels.SquaredDistanceV128(longCodes, shortCodes));
        Assert.Throws<ArgumentException>(() => Int8Kernels.SquaredDistanceV256(longCodes, shortCodes));
        Assert.Throws<ArgumentException>(() => Int8Kernels.SquaredDistanceV512(longCodes, shortCodes));
    }

    [Fact]
    public void BitKernels_RejectMismatchedLengthsInRelease()
    {
        var longBits = new ulong[128];
        var shortBits = new ulong[2];

        Assert.Throws<ArgumentException>(() => BitKernels.HammingV128(longBits, shortBits));
        Assert.Throws<ArgumentException>(() => BitKernels.HammingV256(longBits, shortBits));
        Assert.Throws<ArgumentException>(() => BitKernels.HammingV512(longBits, shortBits));
    }
}
