using Faro;
using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.DistanceTests;

public class CosineEquivalenceTests
{
    public static TheoryData<int> Dimensions { get; } = new()
    {
        1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 767, 768,
    };

    [Theory]
    [MemberData(nameof(Dimensions))]
    public void Cosine_Dispatcher_MatchesScalarReference(int dim)
    {
        var a = DeterministicRandom.GaussianVector(dim, DeterministicRandom.DeriveSeed(0x5eed_1001, dim));
        var b = DeterministicRandom.GaussianVector(dim, DeterministicRandom.DeriveSeed(0x5eed_1002, dim));

        float reference = Distances.CosineScalar(a, b);
        float tolerance = 1e-4f;

        Assert.True(
            MathF.Abs(Distances.Cosine(a, b) - reference) <= tolerance,
            $"cosine dispatcher produced {Distances.Cosine(a, b)}, expected {reference}.");
    }

    [Theory]
    [MemberData(nameof(Dimensions))]
    public void Cosine_OnUnitVectors_IsOneMinusDot(int dim)
    {
        // The pre-normalization fast path: with unit vectors the cosine distance is 1 - dot.
        var a = DeterministicRandom.UnitVector(dim, DeterministicRandom.DeriveSeed(0x5eed_1003, dim));
        var b = DeterministicRandom.UnitVector(dim, DeterministicRandom.DeriveSeed(0x5eed_1004, dim));

        float expected = 1f - Distances.Dot(a, b);
        FloatAssertions.Equal(expected, Distances.Cosine(a, b), 1e-5f);
    }

    [Fact]
    public void Cosine_IdenticalVectors_IsZeroAndNeverNegative()
    {
        var v = DeterministicRandom.GaussianVector(768, 0x5eed_1005);
        float distance = Distances.Cosine(v, v);
        FloatAssertions.Equal(0f, distance, 1e-5f);
        Assert.True(distance >= 0f, $"distance must be clamped to [0, 2], got {distance}.");
    }

    [Fact]
    public void Cosine_OppositeVectors_IsTwo()
    {
        var v = DeterministicRandom.GaussianVector(256, 0x5eed_1006);
        var negated = v.Select(x => -x).ToArray();
        FloatAssertions.Equal(2f, Distances.Cosine(v, negated), 1e-5f);
    }

    [Fact]
    public void Cosine_OrthogonalVectors_IsOne()
    {
        FloatAssertions.Equal(1f, Distances.Cosine([1f, 0f], [0f, 1f]), 1e-6f);
        FloatAssertions.Equal(1f, Distances.Cosine([3f, 0f, 0f], [0f, -7f, 0f]), 1e-6f);
    }

    [Fact]
    public void Cosine_IsInvariantToPerVectorScaling()
    {
        var a = DeterministicRandom.GaussianVector(384, 0x5eed_1007);
        var b = DeterministicRandom.GaussianVector(384, 0x5eed_1008);

        float baseline = Distances.Cosine(a, b);
        float scaledA = Distances.Cosine(a.Select(x => x * 1000f).ToArray(), b);
        float scaledB = Distances.Cosine(a, b.Select(x => x * 0.001f).ToArray());

        FloatAssertions.Equal(baseline, scaledA, 1e-5f);
        FloatAssertions.Equal(baseline, scaledB, 1e-5f);
    }

    [Fact]
    public void Cosine_ZeroVector_IsTreatedAsOrthogonal()
    {
        var v = DeterministicRandom.GaussianVector(64, 0x5eed_1009);
        var zero = new float[64];

        FloatAssertions.Equal(1f, Distances.Cosine(v, zero), 0f);
        FloatAssertions.Equal(1f, Distances.Cosine(zero, v), 0f);
        FloatAssertions.Equal(1f, Distances.Cosine(zero, zero), 0f);
        FloatAssertions.Equal(1f, Distances.CosineScalar(v, zero), 0f);
        FloatAssertions.Equal(1f, Distances.CosineScalar(zero, v), 0f);
    }

    [Fact]
    public void Cosine_RangeIsClamped()
    {
        var v = DeterministicRandom.GaussianVector(512, 0x5eed_100a);
        var w = DeterministicRandom.GaussianVector(512, 0x5eed_100b);
        float distance = Distances.Cosine(v, w);
        Assert.InRange(distance, 0f, 2f);
    }

    [Fact]
    public void Cosine_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => Distances.Cosine(new float[3], new float[4]));
        Assert.Throws<ArgumentException>(() => Distances.CosineScalar(new float[3], new float[4]));
    }
}
