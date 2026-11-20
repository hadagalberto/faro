using Faro.Testing;
using Xunit;

namespace Faro.Tests.Quantization;

/// <summary>
/// End-to-end checks that quantized search returns full-precision scores (proving the rescore
/// pass runs) and recovers the exact neighbours on deterministic clustered data.
/// </summary>
public class QuantizationIntegrationTests
{
    private const int Queries = 12;
    private const int K = 10;

    [Fact]
    public void Flat_Int8_LazyTraining_RescoresAndRecoversNeighbours()
    {
        const int dim = 64;
        var (data, queries) = BuildClustered(600, dim, 30, 0x5eed_1001);
        var (exact, quantized) = BuildFlatPair(dim, Faro.Quantization.Int8(rescore: 6, quantile: 0.99), data);

        double recall = MeasureRecall(exact, quantized, data, queries, dim);
        Assert.True(recall >= 0.85, $"int8 + rescore(6) recall was {recall:F3}.");
    }

    [Fact]
    public void Flat_Binary_LazyTraining_RescoresAndRecoversNeighbours()
    {
        const int dim = 128;
        var (data, queries) = BuildClustered(600, dim, 30, 0x5eed_1002);
        var (exact, quantized) = BuildFlatPair(dim, Faro.Quantization.Binary(rescore: 16), data);

        double recall = MeasureRecall(exact, quantized, data, queries, dim);
        Assert.True(recall >= 0.85, $"binary + rescore(16) recall was {recall:F3}.");
    }

    [Fact]
    public void Flat_Int8_PretrainedCodec_IsUsedImmediately()
    {
        const int dim = 32;
        var (data, queries) = BuildClustered(400, dim, 20, 0x5eed_1003);
        var trained = ScalarQuantizer.Train(data.Take(200), dim, Distance.Cosine, quantile: 0.995);
        var exact = new FlatIndex(new FlatOptions { Dimensions = dim, ParallelThreshold = int.MaxValue });
        var quantized = new FlatIndex(new FlatOptions
        {
            Dimensions = dim,
            ParallelThreshold = int.MaxValue,
            Quantization = Faro.Quantization.Int8(trained, rescore: 4),
        });
        foreach (float[] vector in data)
        {
            exact.Add(vector);
            quantized.Add(vector);
        }

        double recall = MeasureRecall(exact, quantized, data, queries, dim);
        Assert.True(recall >= 0.8, $"pretrained int8 recall was {recall:F3}.");
    }

    [Fact]
    public void Hnsw_Int8AndBinary_ReturnFullPrecisionScoresAndRecoverNeighbours()
    {
        const int dim = 64;
        var (data, queries) = BuildClustered(1_200, dim, 60, 0x5eed_1004);

        var exact = new FlatIndex(new FlatOptions { Dimensions = dim });
        foreach (float[] vector in data)
        {
            exact.Add(vector);
        }

        var int8 = BuildHnsw(dim, Faro.Quantization.Int8(rescore: 4, quantile: 0.99), data);
        var binary = BuildHnsw(dim, Faro.Quantization.Binary(rescore: 8), data);

        double recall8 = MeasureRecall(exact, int8, data, queries, dim);
        double recallBinary = MeasureRecall(exact, binary, data, queries, dim);
        Assert.True(recall8 >= 0.9, $"HNSW int8 recall was {recall8:F3}.");
        Assert.True(recallBinary >= 0.8, $"HNSW binary recall was {recallBinary:F3}.");
    }

    [Fact]
    public void Hnsw_QuantizedSearch_ScoresComeFromFullPrecisionRescore()
    {
        const int dim = 32;
        var (data, queries) = BuildClustered(500, dim, 25, 0x5eed_1005);
        var hnsw = BuildHnsw(dim, Faro.Quantization.Int8(rescore: 4), data);

        foreach (float[] query in queries)
        {
            foreach (SearchResult hit in hnsw.Search(query, K, efSearch: 128))
            {
                float expected = Distances.Cosine(query, data[hit.Id]);
                FloatAssertions.Equal(expected, hit.Score, 1e-4f);
            }
        }
    }

    [Fact]
    public void BinaryQuantization_OnHammingCollections_MatchesExactHamming()
    {
        const int dim = 64;
        var data = Enumerable.Range(0, 300)
            .Select(i => DeterministicRandom.UniformVector(dim, DeterministicRandom.DeriveSeed(0x5eed_1201, i)))
            .ToArray();
        var query = DeterministicRandom.UniformVector(dim, 0x5eed_1202);

        var exact = new FlatIndex(new FlatOptions { Dimensions = dim, Distance = Distance.Hamming });
        var quantized = new FlatIndex(new FlatOptions
        {
            Dimensions = dim,
            Distance = Distance.Hamming,
            Quantization = Faro.Quantization.Binary(rescore: 4),
        });
        var hnsw = new HnswIndex(new HnswOptions
        {
            Dimensions = dim,
            Distance = Distance.Hamming,
            EfSearch = 128,
            Quantization = Faro.Quantization.Binary(rescore: 4),
        });
        foreach (float[] vector in data)
        {
            exact.Add(vector);
            quantized.Add(vector);
            hnsw.Add(vector);
        }

        // A binary code of a Hamming collection is the stored bit pattern itself, so the first
        // pass and the rescore agree exactly and Flat results must be bit-identical to exact.
        Assert.Equal(exact.Search(query, 10), quantized.Search(query, 10));
        foreach (SearchResult hit in hnsw.Search(query, 10, efSearch: 128))
        {
            FloatAssertions.Equal(
                Distances.Hamming(BitVectors.PackSignBits(query), BitVectors.PackSignBits(data[hit.Id])),
                hit.Score,
                1e-4f);
        }
    }

    [Fact]
    public void Int8Quantization_OnDotProductCollections_RescoresToExactDot()
    {
        const int dim = 48;
        var sample = Enumerable.Range(0, 200)
            .Select(i => DeterministicRandom.UniformVector(dim, DeterministicRandom.DeriveSeed(0x5eed_1301, i)))
            .ToArray();
        var trained = ScalarQuantizer.Train(sample, dim, Distance.DotProduct);
        var query = DeterministicRandom.UniformVector(dim, 0x5eed_1302);

        var flat = new FlatIndex(new FlatOptions
        {
            Dimensions = dim,
            Distance = Distance.DotProduct,
            Quantization = Faro.Quantization.Int8(trained, rescore: 4),
        });
        var hnsw = new HnswIndex(new HnswOptions
        {
            Dimensions = dim,
            Distance = Distance.DotProduct,
            EfSearch = 128,
            Quantization = Faro.Quantization.Int8(trained, rescore: 4),
        });
        foreach (float[] vector in sample)
        {
            flat.Add(vector);
            hnsw.Add(vector);
        }

        foreach (SearchResult hit in flat.Search(query, 10).Concat(hnsw.Search(query, 10, efSearch: 128)))
        {
            FloatAssertions.Equal(Distances.Dot(query, sample[hit.Id]), hit.Score, 1e-4f);
        }
    }

    [Fact]
    public void QuantizedIndexes_RejectInvalidInputs()
    {
        Assert.Throws<ArgumentException>(() => new FlatIndex(new FlatOptions
        {
            Dimensions = 8,
            Distance = Distance.Hamming,
            Quantization = Faro.Quantization.Int8(rescore: 4),
        }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Faro.Quantization.Int8(rescore: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Faro.Quantization.Int8(rescore: 4, quantile: 0.1));

        var flatDim8 = new FlatIndex(new FlatOptions { Dimensions = 8, Quantization = Faro.Quantization.Binary() });
        var hnswDim8 = new HnswIndex(new HnswOptions { Dimensions = 8, Quantization = Faro.Quantization.Binary() });

        Assert.Throws<ArgumentException>(() => flatDim8.Add(new float[7]));
        Assert.Throws<ArgumentException>(() => flatDim8.Add(WithNaN(8)));
        Assert.Throws<ArgumentException>(() => flatDim8.Search(new float[7], 1));
        Assert.Throws<ArgumentException>(() => flatDim8.Search(WithNaN(8), 1));
        Assert.Throws<ArgumentException>(() => hnswDim8.Add(new float[7]));
        Assert.Throws<ArgumentException>(() => hnswDim8.Add(WithNaN(8)));
        Assert.Throws<ArgumentException>(() => hnswDim8.Search(new float[7], 1));
        Assert.Throws<ArgumentException>(() => hnswDim8.Search(WithNaN(8), 1));
    }

    [Fact]
    public void PretrainedCodec_MismatchedShape_Throws()
    {
        var trained = ScalarQuantizer.Train([[0f, 1f], [1f, 0f]], 2, Distance.Cosine);

        Assert.Throws<ArgumentException>(() => new FlatIndex(new FlatOptions
        {
            Dimensions = 3,
            Quantization = Faro.Quantization.Int8(trained),
        }));
        Assert.Throws<ArgumentException>(() => new HnswIndex(new HnswOptions
        {
            Dimensions = 2,
            Distance = Distance.Euclidean,
            Quantization = Faro.Quantization.Int8(trained),
        }));
        Assert.Throws<ArgumentException>(() => new FlatIndex(new FlatOptions
        {
            Dimensions = 2,
            Quantization = Faro.Quantization.Binary(BinaryQuantizer.Create(3, Distance.Cosine)),
        }));
    }

    [Fact]
    public void EmptyQuantizedIndexes_SearchReturnsEmpty()
    {
        var flat = new FlatIndex(new FlatOptions { Dimensions = 8, Quantization = Faro.Quantization.Int8() });
        var hnsw = new HnswIndex(new HnswOptions { Dimensions = 8, Quantization = Faro.Quantization.Binary() });

        Assert.Empty(flat.Search(new float[8], 4));
        Assert.Empty(hnsw.Search(new float[8], 4));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuantizedFlatSearch_IsAllocationFreeInSteadyState(bool binary)
    {
        const int dim = 64;
        var (data, _) = BuildClustered(512, dim, 32, 0x5eed_1006);
        var options = new FlatOptions
        {
            Dimensions = dim,
            ParallelThreshold = int.MaxValue,
            Quantization = binary ? Faro.Quantization.Binary(rescore: 8) : Faro.Quantization.Int8(rescore: 4),
        };
        var index = new FlatIndex(options);
        foreach (float[] vector in data)
        {
            index.Add(vector);
        }

        float[] query = data[0];
        var buffer = new SearchResult[10];
        for (int i = 0; i < 2_000; i++)
        {
            index.Search(query, 10, buffer);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 2_000; i++)
        {
            index.Search(query, 10, buffer);
        }

        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuantizedHnswSearch_IsAllocationFreeInSteadyState(bool binary)
    {
        const int dim = 64;
        var (data, _) = BuildClustered(512, dim, 32, 0x5eed_1007);
        var options = new HnswOptions
        {
            Dimensions = dim,
            EfSearch = 64,
            Quantization = binary
                ? Faro.Quantization.Binary(rescore: 8)
                : Faro.Quantization.Int8(rescore: 4),
        };
        var index = new HnswIndex(options);
        foreach (float[] vector in data)
        {
            index.Add(vector);
        }

        float[] query = data[0];
        var buffer = new SearchResult[10];
        for (int i = 0; i < 2_000; i++)
        {
            index.Search(query, 10, buffer);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 2_000; i++)
        {
            index.Search(query, 10, buffer);
        }

        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
    }

    private static (FlatIndex Exact, FlatIndex Quantized) BuildFlatPair(
        int dim, QuantizationOptions quantization, float[][] data)
    {
        var exact = new FlatIndex(new FlatOptions { Dimensions = dim, ParallelThreshold = int.MaxValue });
        var quantized = new FlatIndex(new FlatOptions
        {
            Dimensions = dim,
            ParallelThreshold = int.MaxValue,
            Quantization = quantization,
        });
        foreach (float[] vector in data)
        {
            exact.Add(vector);
            quantized.Add(vector);
        }

        return (exact, quantized);
    }

    private static HnswIndex BuildHnsw(int dim, QuantizationOptions quantization, float[][] data)
    {
        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = dim,
            Distance = Distance.Cosine,
            M = 16,
            EfConstruction = 128,
            EfSearch = 128,
            Seed = 0x5eed_2000,
            Quantization = quantization,
        });
        foreach (float[] vector in data)
        {
            index.Add(vector);
        }

        return index;
    }

    private static double MeasureRecall(
        FlatIndex exact, FlatIndex quantized, float[][] data, float[][] queries, int dim)
    {
        double sum = 0;
        foreach (float[] query in queries)
        {
            var truth = exact.Search(query, K).Select(hit => hit.Id).ToHashSet();
            SearchResult[] hits = quantized.Search(query, K);
            Assert.Equal(K, hits.Length);
            foreach (SearchResult hit in hits)
            {
                FloatAssertions.Equal(Distances.Cosine(query, data[hit.Id]), hit.Score, 1e-4f);
            }

            sum += hits.Count(hit => truth.Contains(hit.Id)) / (double)K;
        }

        return sum / queries.Length;
    }

    private static double MeasureRecall(
        FlatIndex exact, HnswIndex quantized, float[][] data, float[][] queries, int dim)
    {
        double sum = 0;
        foreach (float[] query in queries)
        {
            var truth = exact.Search(query, K).Select(hit => hit.Id).ToHashSet();
            SearchResult[] hits = quantized.Search(query, K, efSearch: 128);
            Assert.Equal(K, hits.Length);
            foreach (SearchResult hit in hits)
            {
                FloatAssertions.Equal(Distances.Cosine(query, data[hit.Id]), hit.Score, 1e-4f);
            }

            sum += hits.Count(hit => truth.Contains(hit.Id)) / (double)K;
        }

        return sum / queries.Length;
    }

    private static float[] WithNaN(int dim)
    {
        var vector = new float[dim];
        vector[0] = float.NaN;
        return vector;
    }

    private static (float[][] Data, float[][] Queries) BuildClustered(
        int count, int dim, int clusters, ulong seed)
    {
        var centroids = new float[clusters][];
        for (int c = 0; c < clusters; c++)
        {
            centroids[c] = DeterministicRandom.GaussianVector(dim, DeterministicRandom.DeriveSeed(seed, c));
        }

        var data = new float[count][];
        ulong noise = seed + 1;
        for (int i = 0; i < count; i++)
        {
            int c = i % clusters;
            data[i] = new float[dim];
            for (int d = 0; d < dim; d++)
            {
                data[i][d] = centroids[c][d] + 0.35f * DeterministicRandom.NextFloat(ref noise);
            }

            Normalize(data[i]);
        }

        var queries = new float[Queries][];
        ulong queryNoise = seed + 2;
        for (int q = 0; q < Queries; q++)
        {
            int c = (int)((long)q * clusters / Queries);
            queries[q] = new float[dim];
            for (int d = 0; d < dim; d++)
            {
                queries[q][d] = centroids[c][d] + 0.2f * DeterministicRandom.NextFloat(ref queryNoise);
            }

            Normalize(queries[q]);
        }

        return (data, queries);
    }

    private static void Normalize(float[] vector)
    {
        double sumSquares = 0;
        foreach (float value in vector)
        {
            sumSquares += (double)value * value;
        }

        float inverse = (float)(1.0 / Math.Sqrt(sumSquares));
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= inverse;
        }
    }
}
