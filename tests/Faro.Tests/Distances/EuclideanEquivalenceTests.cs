using Faro;
using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.DistanceTests;

public class EuclideanEquivalenceTests
{
    public static TheoryData<int> Dimensions { get; } = new()
    {
        1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 767, 768,
    };

    [Theory]
    [MemberData(nameof(Dimensions))]
    public void Euclidean_AllVectorPaths_MatchScalarReference(int dim)
    {
        var a = DeterministicRandom.GaussianVector(dim, DeterministicRandom.DeriveSeed(0x5eed_2001, dim));
        var b = DeterministicRandom.GaussianVector(dim, DeterministicRandom.DeriveSeed(0x5eed_2002, dim));

        float reference = Distances.EuclideanSquaredScalar(a, b);
        float tolerance = 1e-4f * (1f + MathF.Abs(reference));

        AssertWithin(Distances.EuclideanSquared(a, b), reference, tolerance, "dispatcher");
        AssertWithin(VectorKernels.EuclideanSquaredScalar(a, b), reference, tolerance, "scalar-kernel");
        AssertWithin(VectorKernels.EuclideanSquaredV128(a, b), reference, tolerance, "Vector128");
        AssertWithin(VectorKernels.EuclideanSquaredV256(a, b), reference, tolerance, "Vector256");
        AssertWithin(VectorKernels.EuclideanSquaredV512(a, b), reference, tolerance, "Vector512");
    }

    [Theory]
    [MemberData(nameof(Dimensions))]
    public void Euclidean_AllVectorPaths_AreBitExact_ForDyadicValues(int dim)
    {
        // Differences, squares and sums of dyadic values are exact in any association order.
        float[] dyadics = [0.5f, 0.25f, 1f, 2f, 0.125f, 0.0625f, 4f, 0.03125f];

        var a = new float[dim];
        var b = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            a[i] = dyadics[(i * 3 + 2) & 7];
            b[i] = dyadics[(i * 5 + 4) & 7];
        }

        float reference = Distances.EuclideanSquaredScalar(a, b);

        Assert.Equal(reference, Distances.EuclideanSquared(a, b));
        Assert.Equal(reference, VectorKernels.EuclideanSquaredV128(a, b));
        Assert.Equal(reference, VectorKernels.EuclideanSquaredV256(a, b));
        Assert.Equal(reference, VectorKernels.EuclideanSquaredV512(a, b));
    }

    [Fact]
    public void Euclidean_KnownValue_HandComputed()
    {
        // (3,4) to origin: squared L2 = 25.
        FloatAssertions.Equal(25f, Distances.EuclideanSquared([3f, 4f], [0f, 0f]), 1e-6f);
        // (1,1,1) to (2,2,2): squared L2 = 3.
        FloatAssertions.Equal(3f, Distances.EuclideanSquared([1f, 1f, 1f], [2f, 2f, 2f]), 1e-6f);
    }

    [Fact]
    public void Euclidean_IdenticalVectors_IsZero()
    {
        var v = DeterministicRandom.GaussianVector(768, 0x5eed_2003);
        FloatAssertions.Equal(0f, Distances.EuclideanSquared(v, v), 1e-4f);
    }

    [Fact]
    public void Euclidean_IsMonotonicInTrueL2()
    {
        // A point at true distance 3 must have squared distance 9 > 4 (distance 2) > 1.
        var origin = new float[] { 0f, 0f };
        float d1 = Distances.EuclideanSquared(origin, [1f, 0f]);
        float d2 = Distances.EuclideanSquared(origin, [2f, 0f]);
        float d3 = Distances.EuclideanSquared(origin, [3f, 0f]);

        Assert.True(d1 < d2 && d2 < d3, $"expected 1 < 4 < 9, got {d1} < {d2} < {d3}");
    }

    [Fact]
    public void Euclidean_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => Distances.EuclideanSquared(new float[3], new float[4]));
        Assert.Throws<ArgumentException>(() => Distances.EuclideanSquaredScalar(new float[3], new float[4]));
    }

    private static void AssertWithin(float value, float expected, float tolerance, string path)
    {
        Assert.True(
            MathF.Abs(value - expected) <= tolerance,
            $"{path} produced {value}, expected {expected} (tolerance {tolerance}).");
    }
}
