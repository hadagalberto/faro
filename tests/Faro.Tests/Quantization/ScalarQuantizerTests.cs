using Faro.Testing;
using Xunit;

namespace Faro.Tests.Quantization;

public class ScalarQuantizerTests
{
    [Fact]
    public void Train_UsesPerDimensionMinMaxForScale()
    {
        var quantizer = ScalarQuantizer.Train(
            [[1f, -2f], [3f, 4f], [-5f, 0f]], dimensions: 2, metric: Distance.Cosine);

        FloatAssertions.Equal(5f / 127f, quantizer.Scales[0], 1e-6f);
        FloatAssertions.Equal(4f / 127f, quantizer.Scales[1], 1e-6f);
        Assert.Equal(-5f, quantizer.Min[0]);
        Assert.Equal(3f, quantizer.Max[0]);
        Assert.Equal(-2f, quantizer.Min[1]);
        Assert.Equal(4f, quantizer.Max[1]);
        Assert.Equal(2, quantizer.CodeLength);
        Assert.Equal(QuantizationKind.Int8, quantizer.Kind);
        Assert.Equal(Distance.Cosine, quantizer.Metric);
    }

    [Fact]
    public void Train_ZeroDimension_UsesUnitScale()
    {
        var quantizer = ScalarQuantizer.Train([[0f, 7f], [0f, -7f]], dimensions: 2, metric: Distance.Euclidean);

        Assert.Equal(1f, quantizer.Scales[0]);
        Assert.Equal(7f / 127f, quantizer.Scales[1]);
    }

    [Fact]
    public void EncodeDecode_KnownVectors_RoundTripWithinHalfAStep()
    {
        float[][] sample = [
            [1f, -2f], [3f, 4f], [-5f, 0f], [2f, -1f], [0f, 3f], [-1f, -4f],
        ];
        var quantizer = ScalarQuantizer.Train(sample, dimensions: 2, metric: Distance.Cosine);
        var code = new byte[2];
        var decoded = new float[2];

        foreach (float[] vector in sample)
        {
            quantizer.Encode(vector, code);
            quantizer.Decode(code, decoded);
            for (int d = 0; d < 2; d++)
            {
                FloatAssertions.Equal(vector[d], decoded[d], quantizer.Scales[d] * 0.5f + 1e-6f);
            }
        }
    }

    [Fact]
    public void Encode_ClampsOutOfRangeValuesToInt8Extremes()
    {
        var quantizer = ScalarQuantizer.Train([[-1f], [1f]], dimensions: 1, metric: Distance.Cosine);
        var code = new byte[1];

        quantizer.Encode([1000f], code);
        Assert.Equal(unchecked((byte)127), code[0]);
        quantizer.Encode([-1000f], code);
        Assert.Equal(unchecked((byte)-127), code[0]);
        quantizer.Encode([0f], code);
        Assert.Equal(0, code[0]);
    }

    [Fact]
    public void Train_QuantileClipping_DiscardsOutliers()
    {
        var sample = Enumerable.Range(0, 20).Select(i => new[] { i == 19 ? 1000f : 0f }).ToArray();

        var unclipped = ScalarQuantizer.Train(sample, 1, Distance.Cosine, quantile: null);
        var clipped = ScalarQuantizer.Train(sample, 1, Distance.Cosine, quantile: 0.8);

        Assert.Equal(1000f, unclipped.Max[0]);
        Assert.Equal(0f, clipped.Max[0]);
        Assert.True(clipped.Scales[0] < unclipped.Scales[0]);
    }

    [Fact]
    public void Distance_Cosine_KnownCodes_MatchHandComputation()
    {
        var quantizer = ScalarQuantizer.Train(
            [[1f, 0f], [0f, 1f], [-1f, 0f], [0f, -1f]], dimensions: 2, metric: Distance.Cosine);
        var a = new byte[2];
        var b = new byte[2];

        quantizer.Encode([1f, 0f], a);
        quantizer.Encode([0f, 1f], b);
        FloatAssertions.Equal(1f, quantizer.Distance(a, b), 1e-6f);

        quantizer.Encode([1f, 0f], b);
        FloatAssertions.Equal(0f, quantizer.Distance(a, b), 1e-6f);

        quantizer.Encode([-1f, 0f], b);
        FloatAssertions.Equal(2f, quantizer.Distance(a, b), 1e-6f);
    }

    [Fact]
    public void Distance_Cosine_OnRandomUnitVectors_IsCloseToExact()
    {
        const int dim = 128;
        var sample = Enumerable.Range(0, 256)
            .Select(i => DeterministicRandom.UnitVector(dim, DeterministicRandom.DeriveSeed(0x5eed_e001, i)))
            .ToArray();
        var quantizer = ScalarQuantizer.Train(sample, dim, Distance.Cosine, quantile: 0.995);
        float[] query = DeterministicRandom.UnitVector(dim, 0x5eed_e002);
        float[] candidate = DeterministicRandom.UnitVector(dim, 0x5eed_e003);
        var qc = new byte[dim];
        var cc = new byte[dim];
        quantizer.Encode(query, qc);
        quantizer.Encode(candidate, cc);

        float approximate = quantizer.Distance(qc, cc);
        float exact = Distances.Cosine(query, candidate);
        FloatAssertions.Equal(exact, approximate, 0.05f);
    }

    [Fact]
    public void Distance_DotProduct_HigherIsCloser()
    {
        const int dim = 64;
        var sample = Enumerable.Range(0, 128)
            .Select(i => DeterministicRandom.UniformVector(dim, DeterministicRandom.DeriveSeed(0x5eed_e011, i)))
            .ToArray();
        var quantizer = ScalarQuantizer.Train(sample, dim, Distance.DotProduct);
        float[] query = DeterministicRandom.UniformVector(dim, 0x5eed_e012);
        float[] same = (float[])query.Clone();
        float[] opposite = query.Select(x => -x).ToArray();
        var qc = new byte[dim];
        var sc = new byte[dim];
        var oc = new byte[dim];
        quantizer.Encode(query, qc);
        quantizer.Encode(same, sc);
        quantizer.Encode(opposite, oc);

        Assert.True(quantizer.Distance(qc, sc) > quantizer.Distance(qc, oc));
    }

    [Fact]
    public void Distance_Euclidean_LowerIsCloser()
    {
        const int dim = 64;
        var sample = Enumerable.Range(0, 128)
            .Select(i => DeterministicRandom.UniformVector(dim, DeterministicRandom.DeriveSeed(0x5eed_e021, i)))
            .ToArray();
        var quantizer = ScalarQuantizer.Train(sample, dim, Distance.Euclidean);
        float[] query = DeterministicRandom.UniformVector(dim, 0x5eed_e022);
        float[] near = query.Select(x => x + 0.01f).ToArray();
        float[] far = query.Select(x => x + 0.9f).ToArray();
        var qc = new byte[dim];
        var nc = new byte[dim];
        var fc = new byte[dim];
        quantizer.Encode(query, qc);
        quantizer.Encode(near, nc);
        quantizer.Encode(far, fc);

        Assert.True(quantizer.Distance(qc, nc) < quantizer.Distance(qc, fc));
    }

    [Fact]
    public void Train_KnownInvalidInputs_Throw()
    {
        Assert.Throws<ArgumentException>(() => ScalarQuantizer.Train([], 2, Distance.Cosine));
        Assert.Throws<ArgumentException>(() => ScalarQuantizer.Train([[1f, 2f]], 3, Distance.Cosine));
        Assert.Throws<ArgumentException>(() => ScalarQuantizer.Train([[float.NaN, 2f]], 2, Distance.Cosine));
        Assert.Throws<ArgumentException>(() => ScalarQuantizer.Train([[1f, 2f]], 2, Distance.Hamming));
        Assert.Throws<ArgumentOutOfRangeException>(() => ScalarQuantizer.Train([[1f]], 0, Distance.Cosine));
        Assert.Throws<ArgumentOutOfRangeException>(() => ScalarQuantizer.Train([[1f]], 1, Distance.Cosine, 0.4));
        Assert.Throws<ArgumentOutOfRangeException>(() => ScalarQuantizer.Train([[1f]], 1, Distance.Cosine, 1.2));
        Assert.Throws<ArgumentOutOfRangeException>(() => ScalarQuantizer.Train([[1f]], 1, Distance.Cosine, double.NaN));
#pragma warning disable CS8625 // null is intentional: invalid-input contract.
        Assert.Throws<ArgumentNullException>(() => ScalarQuantizer.Train(null!, 1, Distance.Cosine));
#pragma warning restore CS8625
    }

    [Fact]
    public void EncodeDecodeDistance_KnownInvalidBuffers_Throw()
    {
        var quantizer = ScalarQuantizer.Train([[-1f], [1f]], 1, Distance.Cosine);
        var code = new byte[1];

        Assert.Throws<ArgumentException>(() => quantizer.Encode(new[] { 1f, 2f }, code));
        Assert.Throws<ArgumentException>(() => quantizer.Encode(new[] { float.NaN }, code));
        Assert.Throws<ArgumentException>(() => quantizer.Encode([1f], Span<byte>.Empty));
        Assert.Throws<ArgumentException>(() => quantizer.Decode(Span<byte>.Empty, new float[1]));
        Assert.Throws<ArgumentException>(() => quantizer.Decode(code, Span<float>.Empty));
        Assert.Throws<ArgumentException>(() => quantizer.Distance(Span<byte>.Empty, code));
        Assert.Throws<NotSupportedException>(() => quantizer.EncodePacked(new ulong[1], code));
    }
}
