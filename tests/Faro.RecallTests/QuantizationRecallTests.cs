using System.Diagnostics;
using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.RecallTests;

/// <summary>
/// Recall gates from SPEC for quantized search on 768-dimensional normalized cluster-gaussian
/// data. Flat (full precision) is the ground truth. Rescoring is asserted with the specified
/// oversampling factors: int8 with 4, binary with 8.
/// </summary>
public class QuantizationRecallTests
{
    private const int Count = 8_000;
    private const int Dimensions = 768;
    private const int Clusters = 100;
    private const int Queries = 25;
    private const int K = 10;

    private readonly ITestOutputHelper _output;

    public QuantizationRecallTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Int8WithRescore_RecallAt10_AtLeast095()
    {
        var fixture = QuantizationFixture.Shared;
        ScalarQuantizer scalar = fixture.Scalar;
        var flatExact = fixture.FlatExact;

        var flat8 = BuildFlat(fixture, Quantization.Int8(scalar, rescore: 4));
        var hnsw8 = BuildHnsw(fixture, Quantization.Int8(scalar, rescore: 4), efSearch: 256);

        double flatRecall = Measure(flatExact, q => flat8.Search(q, K), fixture.Queries);
        double hnswRecall = Measure(flatExact, q => hnsw8.Search(q, K, efSearch: 256), fixture.Queries);

        Print("int8 + rescore(4)", flatRecall, hnswRecall);

        Assert.True(flatRecall >= 0.95, $"Flat int8 + rescore(4) recall@10 was {flatRecall:F3}, below 0.95.");
        Assert.True(hnswRecall >= 0.95, $"HNSW int8 + rescore(4) recall@10 was {hnswRecall:F3}, below 0.95.");
    }

    [Fact]
    public void BinaryWithRescore8_RecallAt10_AtLeast085()
    {
        var fixture = QuantizationFixture.Shared;
        var binary = BinaryQuantizer.Create(Dimensions, Distance.Cosine);
        var flatExact = fixture.FlatExact;

        var flatBinary = BuildFlat(fixture, Quantization.Binary(binary, rescore: 8));
        var hnswBinary = BuildHnsw(fixture, Quantization.Binary(binary, rescore: 8), efSearch: 512);

        double flatRecall = Measure(flatExact, q => flatBinary.Search(q, K), fixture.Queries);
        double hnswRecall = Measure(flatExact, q => hnswBinary.Search(q, K, efSearch: 512), fixture.Queries);

        Print("binary + rescore(8)", flatRecall, hnswRecall);

        Assert.True(flatRecall >= 0.85, $"Flat binary + rescore(8) recall@10 was {flatRecall:F3}, below 0.85.");
        Assert.True(hnswRecall >= 0.85, $"HNSW binary + rescore(8) recall@10 was {hnswRecall:F3}, below 0.85.");
    }

    private static FlatIndex BuildFlat(QuantizationFixture fixture, QuantizationOptions quantization)
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = Dimensions, Quantization = quantization });
        foreach (float[] vector in fixture.Data)
        {
            index.Add(vector);
        }

        return index;
    }

    private static HnswIndex BuildHnsw(QuantizationFixture fixture, QuantizationOptions quantization, int efSearch)
    {
        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = Dimensions,
            Distance = Distance.Cosine,
            M = 16,
            EfConstruction = 128,
            EfSearch = efSearch,
            Seed = 0x5eed_9000,
            Quantization = quantization,
        });
        foreach (float[] vector in fixture.Data)
        {
            index.Add(vector);
        }

        return index;
    }

    private static double Measure(FlatIndex exact, Func<float[], SearchResult[]> search, float[][] queries)
    {
        double sum = 0;
        foreach (float[] query in queries)
        {
            var truth = exact.Search(query, K).Select(hit => hit.Id).ToHashSet();
            sum += search(query).Count(hit => truth.Contains(hit.Id)) / (double)K;
        }

        return sum / queries.Length;
    }

    private void Print(string configuration, double flatRecall, double hnswRecall)
    {
        _output.WriteLine("| configuration        | N     | d   | k  | flat recall@10 | hnsw recall@10 |");
        _output.WriteLine("|----------------------|-------|-----|----|----------------|----------------|");
        _output.WriteLine($"| {configuration,-20} | {Count,5} | {Dimensions,3} | {K,2} | {flatRecall,14:F3} | {hnswRecall,14:F3} |");
    }

    /// <summary>Built once per process: generating and indexing 8k x 768 vectors is the expensive part.</summary>
    private sealed class QuantizationFixture
    {
        private static readonly Lazy<QuantizationFixture> Lazy = new(Build, isThreadSafe: true);

        internal static QuantizationFixture Shared => Lazy.Value;

        internal float[][] Data { get; init; } = [];
        internal float[][] Queries { get; init; } = [];
        internal ScalarQuantizer Scalar { get; init; } = null!;
        internal FlatIndex FlatExact { get; init; } = null!;

        private static QuantizationFixture Build()
        {
            var centroids = new float[Clusters][];
            for (int c = 0; c < Clusters; c++)
            {
                centroids[c] = DeterministicRandom.GaussianVector(Dimensions, DeterministicRandom.DeriveSeed(0x5eed_9001, c));
            }

            var data = new float[Count][];
            ulong noise = 0x5eed_9002;
            for (int i = 0; i < Count; i++)
            {
                float[] centroid = centroids[i % Clusters];
                var vector = new float[Dimensions];
                for (int d = 0; d < Dimensions; d++)
                {
                    vector[d] = centroid[d] + 0.05f * DeterministicRandom.NextFloat(ref noise);
                }

                Normalize(vector);
                data[i] = vector;
            }

            var queries = new float[QuantizationRecallTests.Queries][];
            ulong queryNoise = 0x5eed_9003;
            for (int q = 0; q < QuantizationRecallTests.Queries; q++)
            {
                float[] centroid = centroids[(int)((long)q * Clusters / QuantizationRecallTests.Queries)];
                var vector = new float[Dimensions];
                for (int d = 0; d < Dimensions; d++)
                {
                    vector[d] = centroid[d] + 0.02f * DeterministicRandom.NextFloat(ref queryNoise);
                }

                Normalize(vector);
                queries[q] = vector;
            }

            var scalar = ScalarQuantizer.Train(data.Take(2048), Dimensions, Distance.Cosine, quantile: 0.995);
            var flat = new FlatIndex(new FlatOptions { Dimensions = Dimensions });
            var watch = Stopwatch.StartNew();
            foreach (float[] vector in data)
            {
                flat.Add(vector);
            }

            Console.WriteLine($"quantization recall fixture: built {Count:N0} x {Dimensions} flat vectors in {watch.Elapsed.TotalSeconds:F1}s");

            return new QuantizationFixture { Data = data, Queries = queries, Scalar = scalar, FlatExact = flat };
        }

        private static void Normalize(float[] vector)
        {
            double sumSquares = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sumSquares += (double)vector[i] * vector[i];
            }

            float inverse = (float)(1.0 / Math.Sqrt(sumSquares));
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] *= inverse;
            }
        }
    }
}
