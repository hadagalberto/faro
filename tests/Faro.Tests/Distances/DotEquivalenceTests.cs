using Faro;
using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.DistanceTests;

public class DotEquivalenceTests
{
    public static TheoryData<int> Dimensions { get; } = new()
    {
        1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 767, 768,
    };

    public static TheoryData<int> KernelDimensions { get; } = new()
    {
        1, 3, 4, 5, 8, 12, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 200, 255, 256, 257, 512, 513, 768,
    };

    [Theory]
    [MemberData(nameof(Dimensions))]
    public void Dot_AllVectorPaths_MatchScalarReference(int dim)
    {
        var a = DeterministicRandom.GaussianVector(dim, DeterministicRandom.DeriveSeed(0x5eed_0001, dim));
        var b = DeterministicRandom.GaussianVector(dim, DeterministicRandom.DeriveSeed(0x5eed_0002, dim));

        float reference = Distances.DotScalar(a, b);
        float tolerance = 1e-4f * (1f + MathF.Abs(reference));

        AssertWithin(Distances.Dot(a, b), reference, tolerance, "dispatcher");
        AssertWithin(VectorKernels.DotScalar(a, b), reference, tolerance, "scalar-kernel");
        AssertWithin(VectorKernels.DotV128(a, b), reference, tolerance, "Vector128");
        AssertWithin(VectorKernels.DotV256(a, b), reference, tolerance, "Vector256");
        AssertWithin(VectorKernels.DotV512(a, b), reference, tolerance, "Vector512");
    }

    [Theory]
    [MemberData(nameof(KernelDimensions))]
    public void Dot_AllVectorPaths_AreBitExact_ForDyadicValues(int dim)
    {
        // Powers of two are summed exactly in floating point under any association order, so
        // every kernel must agree with the reference bit for bit on dyadic inputs.
        float[] dyadics = [0.5f, 0.25f, 1f, 2f, 0.125f, 4f, -0.5f, -2f];

        var a = new float[dim];
        var b = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            a[i] = dyadics[(i * 7 + 3) & 7];
            b[i] = dyadics[(i * 5 + 1) & 7];
        }

        float reference = Distances.DotScalar(a, b);

        Assert.Equal(reference, Distances.Dot(a, b));
        Assert.Equal(reference, VectorKernels.DotScalar(a, b));
        Assert.Equal(reference, VectorKernels.DotV128(a, b));
        Assert.Equal(reference, VectorKernels.DotV256(a, b));
        Assert.Equal(reference, VectorKernels.DotV512(a, b));
    }

    [Theory]
    [MemberData(nameof(KernelDimensions))]
    public void SumSquares_AllVectorPaths_MatchScalarReference(int dim)
    {
        var v = DeterministicRandom.GaussianVector(dim, DeterministicRandom.DeriveSeed(0x5eed_0003, dim));

        float reference = VectorKernels.SumSquaresScalar(v);
        float tolerance = 1e-4f * (1f + MathF.Abs(reference));

        AssertWithin(VectorKernels.SumSquares(v), reference, tolerance, "dispatcher");
        AssertWithin(VectorKernels.SumSquaresV128(v), reference, tolerance, "Vector128");
        AssertWithin(VectorKernels.SumSquaresV256(v), reference, tolerance, "Vector256");
        AssertWithin(VectorKernels.SumSquaresV512(v), reference, tolerance, "Vector512");
    }

    [Fact]
    public void Dot_KnownValues_HandComputed()
    {
        FloatAssertions.Equal(12f, Distances.Dot([1f, 2f, 3f], [4f, -5f, 6f]), 1e-6f);
        FloatAssertions.Equal(0.5f, Distances.Dot([0.5f, 0.5f], [0.5f, 0.5f]), 1e-6f);
        FloatAssertions.Equal(10f, Distances.Dot([1f, 1, 1, 1], [10f, 0, 0, 0]), 1e-6f);
    }

    [Fact]
    public void Dot_OrthogonalVectors_IsZero()
    {
        FloatAssertions.Equal(0f, Distances.Dot([1f, 0f], [0f, 1f]), 1e-6f);
        FloatAssertions.Equal(0f, Distances.Dot([3f, 0f, 0f], [0f, -7f, 0f]), 1e-6f);
    }

    [Fact]
    public void Dot_EmptyVectors_ReturnZero()
    {
        FloatAssertions.Equal(0f, Distances.Dot([], []), 0f);
    }

    [Fact]
    public void Dot_SameVector_EqualsSquaredNorm()
    {
        var v = DeterministicRandom.GaussianVector(768, 0x5eed_0004);
        float squaredNorm = Distances.Dot(v, v);
        FloatAssertions.Equal(squaredNorm, VectorKernels.SumSquares(v), 1e-3f * (1f + MathF.Abs(squaredNorm)));
    }

    [Fact]
    public void Dot_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => Distances.Dot(new float[3], new float[4]));
        Assert.Throws<ArgumentException>(() => Distances.DotScalar(new float[7], new float[6]));
    }

    private static void AssertWithin(float value, float expected, float tolerance, string path)
    {
        Assert.True(
            MathF.Abs(value - expected) <= tolerance,
            $"{path} produced {value}, expected {expected} (tolerance {tolerance}).");
    }
}
