using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Quantization;

/// <summary>
/// Every int8 SIMD path must return the exact integer result of the scalar reference, including
/// the scalar tail for lengths that are not a multiple of a SIMD register.
/// </summary>
public class Int8KernelTests
{
    public static TheoryData<int> Lengths => new()
    {
        1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 768,
    };

    [Theory]
    [MemberData(nameof(Lengths))]
    public void Dot_AllPathsAgreeWithScalar(int length)
    {
        var (a, b) = MakeCodes(length, 0x5eed_d001);

        int expected = Int8Kernels.DotScalar(a, b);
        Assert.Equal(expected, Int8Kernels.Dot(a, b));
        Assert.Equal(expected, Int8Kernels.DotV128(a, b));
        Assert.Equal(expected, Int8Kernels.DotV256(a, b));
        Assert.Equal(expected, Int8Kernels.DotV512(a, b));
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void SquaredDistance_AllPathsAgreeWithScalar(int length)
    {
        var (a, b) = MakeCodes(length, 0x5eed_d002);

        int expected = Int8Kernels.SquaredDistanceScalar(a, b);
        Assert.Equal(expected, Int8Kernels.SquaredDistance(a, b));
        Assert.Equal(expected, Int8Kernels.SquaredDistanceV128(a, b));
        Assert.Equal(expected, Int8Kernels.SquaredDistanceV256(a, b));
        Assert.Equal(expected, Int8Kernels.SquaredDistanceV512(a, b));
    }

    [Fact]
    public void Dot_KnownVector_MatchesHandComputation()
    {
        sbyte[] a = [1, 2, -3, 4, 127, -127];
        sbyte[] b = [5, -6, 7, 8, -1, -1];

        // 1*5 + 2*-6 + -3*7 + 4*8 + 127*-1 + -127*-1 = 5-12-21+32-127+127 = 4
        Assert.Equal(4, Int8Kernels.Dot(a, b));
        Assert.Equal(4, Int8Kernels.DotScalar(a, b));
    }

    [Fact]
    public void SquaredDistance_KnownVector_MatchesHandComputation()
    {
        sbyte[] a = [1, 2, -3, 4, 127, -127];
        sbyte[] b = [5, -6, 7, 8, -1, -1];

        // 16 + 64 + 100 + 16 + 128^2 + 126^2 = 196 + 16384 + 15876 = 32456
        Assert.Equal(32_456, Int8Kernels.SquaredDistance(a, b));
        Assert.Equal(32_456, Int8Kernels.SquaredDistanceScalar(a, b));
    }

    [Fact]
    public void Dot_MaximumMagnitudeCodes_DoNotOverflowInt()
    {
        // 768 * 127^2 = 12,387,072 — well inside int, but worth pinning down.
        var a = Enumerable.Repeat((sbyte)127, 768).ToArray();
        var b = Enumerable.Repeat((sbyte)-127, 768).ToArray();

        Assert.Equal(-12_387_072, Int8Kernels.Dot(a, b));
    }

    private static (sbyte[] A, sbyte[] B) MakeCodes(int length, ulong seed)
    {
        ulong state = seed;
        var a = new sbyte[length];
        var b = new sbyte[length];
        for (int i = 0; i < length; i++)
        {
            a[i] = (sbyte)((int)(DeterministicRandom.Next(ref state) % 255) - 127);
            b[i] = (sbyte)((int)(DeterministicRandom.Next(ref state) % 255) - 127);
        }

        return (a, b);
    }
}
