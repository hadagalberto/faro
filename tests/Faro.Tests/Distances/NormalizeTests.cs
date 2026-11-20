using Faro;
using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.DistanceTests;

public class NormalizeTests
{
    [Fact]
    public void Normalize_KnownVector_ReturnsOriginalNorm()
    {
        var vector = new float[] { 3f, 4f };
        float norm = Distances.Normalize(vector);

        FloatAssertions.Equal(5f, norm, 1e-6f);
        FloatAssertions.Equal(0.6f, vector[0], 1e-6f);
        FloatAssertions.Equal(0.8f, vector[1], 1e-6f);
    }

    [Fact]
    public void Normalize_ProducesUnitLength()
    {
        for (int dim = 1; dim <= 33; dim++)
        {
            var vector = DeterministicRandom.GaussianVector(dim, DeterministicRandom.DeriveSeed(0x5eed_4001, dim));
            Distances.Normalize(vector);

            FloatAssertions.Equal(1f, Distances.Norm(vector), 1e-5f);
        }
    }

    [Fact]
    public void Normalize_PreservesDirection()
    {
        var original = DeterministicRandom.GaussianVector(256, 0x5eed_4002);
        var normalized = (float[])original.Clone();
        Distances.Normalize(normalized);

        FloatAssertions.Equal(0f, Distances.Cosine(original, normalized), 1e-5f);
    }

    [Fact]
    public void Normalize_ZeroVector_IsNoOp()
    {
        var vector = new float[16];
        FloatAssertions.Equal(0f, Distances.Normalize(vector), 0f);
        Assert.All(vector, x => Assert.Equal(0f, x));
    }

    [Fact]
    public void Norm_EmptyVector_IsZero()
    {
        FloatAssertions.Equal(0f, Distances.Norm([]), 0f);
    }

    [Fact]
    public void CopyNormalized_MatchesInPlaceNormalize()
    {
        var source = DeterministicRandom.GaussianVector(128, 0x5eed_4003);
        var inPlace = (float[])source.Clone();
        Distances.Normalize(inPlace);

        var copied = new float[128];
        Distances.CopyNormalized(source, copied);

        Assert.Equal(inPlace, copied);
        // The source must remain untouched: it must equal freshly generated data.
        Assert.Equal(DeterministicRandom.GaussianVector(128, 0x5eed_4003), source);
    }

    [Fact]
    public void CopyNormalized_ZeroSource_IsCopiedVerbatim()
    {
        var source = new float[8];
        var destination = new float[8];
        Distances.CopyNormalized(source, destination);
        Assert.Equal(source, destination);
    }

    [Fact]
    public void CopyNormalized_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => Distances.CopyNormalized(new float[3], new float[4]));
    }

    [Fact]
    public void LowerIsCloser_MatchesMetricOrdering()
    {
        Assert.True(Distances.LowerIsCloser(Distance.Cosine));
        Assert.True(Distances.LowerIsCloser(Distance.Euclidean));
        Assert.True(Distances.LowerIsCloser(Distance.Hamming));
        Assert.False(Distances.LowerIsCloser(Distance.DotProduct));
    }
}
