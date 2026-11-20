using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Hnsw;

public class HnswIndexTests
{
    private const int Dim = 32;

    private static float[] Vector(int id, ulong seed = 0x5eed_e000) =>
        DeterministicRandom.GaussianVector(Dim, DeterministicRandom.DeriveSeed(seed, id));

    private static HnswIndex CreateIndex(Distance distance, int m = 8, int efC = 120, int efS = 64) =>
        new(new HnswOptions
        {
            Dimensions = Dim,
            Distance = distance,
            NormalizeOnAdd = false,
            M = m,
            EfConstruction = efC,
            EfSearch = efS,
            Seed = 0x1234_5678,
        });

    // ---- exactness with a beam wide enough to visit the whole graph ----

    [Theory]
    [InlineData(Distance.Cosine)]
    [InlineData(Distance.DotProduct)]
    [InlineData(Distance.Euclidean)]
    public void Search_WideBeam_MatchesFlatExactly(Distance distance)
    {
        var hnsw = CreateIndex(distance, efC: 400, efS: 400);
        var flat = new FlatIndex(new FlatOptions
        {
            Dimensions = Dim,
            Distance = distance,
            NormalizeOnAdd = false,
            ParallelThreshold = int.MaxValue,
        });

        var data = new List<float[]>();
        for (int id = 0; id < 250; id++)
        {
            var vector = Vector(id);
            data.Add(vector);
            flat.Add(vector);
            hnsw.Add(vector);
        }

        for (int q = 0; q < 8; q++)
        {
            var query = Vector(10_000 + q);
            var expected = flat.Search(query, 10);
            var actual = hnsw.Search(query, 10, efSearch: 400);

            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i].Id, actual[i].Id);
                FloatAssertions.Equal(expected[i].Score, actual[i].Score, 1e-4f);
            }
        }
    }

    [Fact]
    public void Search_Hamming_WideBeam_MatchesFlatExactly()
    {
        const int dimensions = 64;
        var hnsw = new HnswIndex(new HnswOptions
        {
            Dimensions = dimensions,
            Distance = Distance.Hamming,
            M = 8,
            EfConstruction = 400,
            EfSearch = 400,
            Seed = 0x1234_5678,
        });
        var flat = new FlatIndex(new FlatOptions
        {
            Dimensions = dimensions,
            Distance = Distance.Hamming,
            ParallelThreshold = int.MaxValue,
        });

        var data = new List<float[]>();
        for (int id = 0; id < 200; id++)
        {
            var vector = DeterministicRandom.UniformVector(dimensions, DeterministicRandom.DeriveSeed(0x5eed_e001, id));
            data.Add(vector);
            flat.Add(vector);
            hnsw.Add(vector);
        }

        for (int q = 0; q < 6; q++)
        {
            var query = DeterministicRandom.UniformVector(dimensions, DeterministicRandom.DeriveSeed(0x5eed_e002, 10_000 + q));
            var expected = flat.Search(query, 10);
            var actual = hnsw.Search(query, 10, efSearch: 400);

            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i].Id, actual[i].Id);
                FloatAssertions.Equal(expected[i].Score, actual[i].Score, 0f);
            }
        }
    }

    // ---- determinism ----

    [Fact]
    public void SameSeedAndOrder_ProduceIdenticalGraphsAndResults()
    {
        static HnswIndex Build()
        {
            var index = new HnswIndex(new HnswOptions
            {
                Dimensions = Dim,
                Distance = Distance.Euclidean,
                NormalizeOnAdd = false,
                M = 6,
                EfConstruction = 100,
                EfSearch = 32,
                Seed = 99,
            });
            for (int id = 0; id < 200; id++)
            {
                index.Add(Vector(id, 0x5eed_e003));
            }

            return index;
        }

        var first = Build();
        var second = Build();

        Assert.Equal(first.MaxLevel, second.MaxLevel);
        var a = first.CurrentView;
        var b = second.CurrentView;
        for (int id = 0; id < a.Count; id++)
        {
            Assert.Equal(a.Nodes[id]!.Level, b.Nodes[id]!.Level);
            for (int layer = 0; layer <= a.Nodes[id]!.Level; layer++)
            {
                Assert.Equal(a.Nodes[id]!.GetLinks(layer), b.Nodes[id]!.GetLinks(layer));
            }
        }

        for (int q = 0; q < 5; q++)
        {
            Assert.Equal(first.Search(Vector(10_000 + q), 10), second.Search(Vector(10_000 + q), 10));
        }
    }

    [Fact]
    public void DifferentSeeds_ProduceDifferentGraphLevels()
    {
        static HnswIndex Build(long seed)
        {
            var index = new HnswIndex(new HnswOptions
            {
                Dimensions = Dim,
                M = 8,
                EfConstruction = 80,
                Seed = seed,
            });
            for (int id = 0; id < 100; id++)
            {
                index.Add(Vector(id));
            }

            return index;
        }

        var first = Build(1);
        var second = Build(2);
        var a = first.CurrentView;
        var b = second.CurrentView;
        bool anyDifferent = false;
        for (int id = 0; id < a.Count; id++)
        {
            if (a.Nodes[id]!.Level != b.Nodes[id]!.Level)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "different seeds must assign different levels");
    }

    // ---- k semantics, buffer overload, storage ----

    [Fact]
    public void Search_EmptyIndex_ReturnsNoHits()
    {
        var index = CreateIndex(Distance.Cosine);
        Assert.Empty(index.Search(Vector(0), 5));

        var buffer = new SearchResult[8];
        Assert.Equal(0, index.Search(Vector(0), 5, buffer));
    }

    [Fact]
    public void Search_KGreaterThanLiveCount_ReturnsAllLiveNodes()
    {
        var index = CreateIndex(Distance.Euclidean);
        for (int id = 0; id < 30; id++)
        {
            index.Add(Vector(id));
        }

        Assert.Equal(30, index.Search(Vector(999), 100).Length);
        Assert.Equal(30, index.Search(Vector(999), 100, new SearchResult[100]));
    }

    [Fact]
    public void Search_KEqualsOne_ReturnsNearest()
    {
        var index = CreateIndex(Distance.Euclidean, efS: 128);
        for (int id = 0; id < 100; id++)
        {
            index.Add(Vector(id));
        }

        var results = index.Search(Vector(0), 1);
        Assert.Single(results);
        Assert.Equal(0, results[0].Id);
        FloatAssertions.Equal(0f, results[0].Score, 1e-4f);
    }

    [Fact]
    public void Search_BufferShorterThanK_ReturnsOnlyTheBest()
    {
        var index = CreateIndex(Distance.Euclidean);
        for (int id = 0; id < 100; id++)
        {
            index.Add(Vector(id));
        }

        var full = index.Search(Vector(123), 10);
        var buffer = new SearchResult[3];
        int written = index.Search(Vector(123), 10, buffer);

        Assert.Equal(3, written);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(full[i].Id, buffer[i].Id);
        }
    }

    [Fact]
    public void Search_EmptyBuffer_ReturnsZero()
    {
        var index = CreateIndex(Distance.Cosine);
        index.Add(Vector(0));
        Assert.Equal(0, index.Search(Vector(1), 5, Span<SearchResult>.Empty));
    }

    [Fact]
    public void Add_ReturnsSequentialStableIds_AndStoresVectors()
    {
        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = Dim,
            NormalizeOnAdd = false,
            InitialCapacity = 2,
            Seed = 7,
        });
        var stored = new float[Dim];
        for (int id = 0; id < 100; id++)
        {
            var vector = Vector(id, 0x5eed_e004);
            Assert.Equal(id, index.Add(vector));
        }

        Assert.Equal(100, index.Count);
        Assert.Equal(100, index.LiveCount);
        for (int id = 0; id < 100; id++)
        {
            Assert.True(index.TryGetVector(id, stored));
            Assert.Equal(Vector(id, 0x5eed_e004), stored);
        }
    }

    [Fact]
    public void TryGetVector_CosineDefault_NormalizesStorage()
    {
        var index = new HnswIndex(new HnswOptions { Dimensions = 4 });
        index.Add([3f, 0f, 0f, 0f]);

        var stored = new float[4];
        Assert.True(index.TryGetVector(0, stored));
        FloatAssertions.Equal(1f, Distances.Norm(stored), 1e-6f);
        Assert.Equal([1f, 0f, 0f, 0f], stored);
    }

    [Fact]
    public void TryGetVector_Hamming_ReconstructsSigns()
    {
        var index = new HnswIndex(new HnswOptions { Dimensions = 3, Distance = Distance.Hamming });
        index.Add([-1f, 2f, -3f]);

        var stored = new float[3];
        Assert.True(index.TryGetVector(0, stored));
        Assert.Equal([-1f, 1f, -1f], stored);
    }

    [Fact]
    public void TryGetVector_UnknownId_ReturnsFalse()
    {
        var index = CreateIndex(Distance.Cosine);
        index.Add(Vector(0));
        Assert.False(index.TryGetVector(1, new float[Dim]));
        Assert.False(index.TryGetVector(500, new float[Dim]));
    }

    [Fact]
    public void Index_ExposesConfiguration()
    {
        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = 16,
            Distance = Distance.DotProduct,
            M = 10,
            EfConstruction = 33,
            EfSearch = 17,
        });

        Assert.Equal(16, index.Dimensions);
        Assert.Equal(Distance.DotProduct, index.Distance);
        Assert.Equal(0, index.Count);
        Assert.Equal(-1, index.MaxLevel);
        Assert.Equal(-1, index.EntryPointId);
    }

    // ---- validation ----

    [Fact]
    public void Constructor_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new HnswIndex(null!));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_NonPositiveDimensions_Throws(int dimensions)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswIndex(new HnswOptions { Dimensions = dimensions }));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_BadInitialCapacity_Throws(int capacity)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswIndex(new HnswOptions { Dimensions = 8, InitialCapacity = capacity }));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Constructor_MBelowTwo_Throws(int m)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswIndex(new HnswOptions { Dimensions = 8, M = m }));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_BadEfConstruction_Throws(int ef)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswIndex(new HnswOptions { Dimensions = 8, EfConstruction = ef }));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_BadEfSearch_Throws(int ef)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswIndex(new HnswOptions { Dimensions = 8, EfSearch = ef }));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Constructor_BadLevelMultiplier_Throws(double multiplier)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswIndex(new HnswOptions { Dimensions = 8, LevelMultiplier = multiplier }));
    }

    [Theory]
    [InlineData(0.5)]
    [InlineData(double.NaN)]
    public void Constructor_BadMc_Throws(double mc)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswIndex(new HnswOptions { Dimensions = 8, Mc = mc }));
    }

    [Fact]
    public void Constructor_NormalizationOnlyAllowedForCosine()
    {
        Assert.Throws<ArgumentException>(() => new HnswIndex(new HnswOptions { Dimensions = 8, Distance = Distance.Euclidean, NormalizeOnAdd = true }));
        Assert.Throws<ArgumentException>(() => new HnswIndex(new HnswOptions { Dimensions = 8, Distance = Distance.DotProduct, NormalizeOnAdd = true }));
        Assert.Throws<ArgumentException>(() => new HnswIndex(new HnswOptions { Dimensions = 8, Distance = Distance.Hamming, NormalizeOnAdd = true }));
    }

    [Fact]
    public void Add_WrongDimensionsOrNonFinite_Throw()
    {
        var index = CreateIndex(Distance.Cosine);
        Assert.Throws<ArgumentException>(() => index.Add(new float[Dim - 1]));

        var poison = new float[Dim];
        poison[3] = float.NaN;
        Assert.Throws<ArgumentException>(() => index.Add(poison));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Search_NonPositiveK_Throws(int k)
    {
        var index = CreateIndex(Distance.Cosine);
        index.Add(Vector(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(Vector(1), k));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(Vector(1), k, new SearchResult[4]));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Search_NonPositiveEf_Throws(int ef)
    {
        var index = CreateIndex(Distance.Cosine);
        index.Add(Vector(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(Vector(1), 3, ef));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(Vector(1), 3, new SearchResult[4], ef));
    }

    [Fact]
    public void Search_WrongDimensionsOrNonFiniteQuery_Throws()
    {
        var index = CreateIndex(Distance.Cosine);
        index.Add(Vector(0));
        Assert.Throws<ArgumentException>(() => index.Search(new float[Dim - 1], 3));

        var poison = new float[Dim];
        poison[2] = float.PositiveInfinity;
        Assert.Throws<ArgumentException>(() => index.Search(poison, 3));
    }

    [Fact]
    public void Delete_NegativeId_Throws()
    {
        var index = CreateIndex(Distance.Cosine);
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Delete(-1));
    }
}
