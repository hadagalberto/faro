using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Flat;

public class FlatIndexTests
{
    private const int Dim = 64;

    private static float[] Vector(int id, int dimensions = Dim, ulong seed = 0x5eed_8000) =>
        DeterministicRandom.GaussianVector(dimensions, DeterministicRandom.DeriveSeed(seed, id));

    // ---- Exactness against the public kernels (bit-identical scores, exact ordering) ----

    [Theory]
    [InlineData(Distance.Cosine)]
    [InlineData(Distance.DotProduct)]
    [InlineData(Distance.Euclidean)]
    public void Search_IsExactAgainstPublicKernel(Distance distance)
    {
        // No normalization: the index scores with the same public kernel the reference uses, so
        // ordering must match exactly, ties included.
        var index = new FlatIndex(new FlatOptions { Dimensions = Dim, Distance = distance, NormalizeOnAdd = false, ParallelThreshold = int.MaxValue });
        var data = new List<float[]>();
        for (int id = 0; id < 300; id++)
        {
            var vector = Vector(id);
            data.Add(vector);
            index.Add(vector);
        }

        for (int q = 0; q < 5; q++)
        {
            var query = Vector(10_000 + q);
            var results = index.Search(query, 10);

            var expected = data.Select((vector, id) => (Id: id, Score: Score(distance, query, vector)))
                .OrderBy(x => Distances.LowerIsCloser(distance) ? x.Score : -x.Score)
                .ThenBy(x => x.Id)
                .Take(10)
                .ToList();

            Assert.Equal(10, results.Length);
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(expected[i].Id, results[i].Id);
                FloatAssertions.Equal(expected[i].Score, results[i].Score, 0f);
            }
        }

        static float Score(Distance distance, ReadOnlySpan<float> query, ReadOnlySpan<float> vector) => distance switch
        {
            Distance.Cosine => Distances.Cosine(query, vector),
            Distance.DotProduct => Distances.Dot(query, vector),
            Distance.Euclidean => Distances.EuclideanSquared(query, vector),
            _ => throw new InvalidOperationException(),
        };
    }

    [Fact]
    public void Search_NormalizedCosine_IsExactAgainstOneMinusDot()
    {
        // Default options: cosine with normalization on. The index then ranks by 1 - dot of
        // unit vectors; the reference recomputes the stored form via TryGetVector and a
        // normalized query, and must agree bit for bit.
        var index = new FlatIndex(new FlatOptions { Dimensions = Dim, ParallelThreshold = int.MaxValue });
        var data = new List<float[]>();
        for (int id = 0; id < 300; id++)
        {
            var vector = Vector(id);
            data.Add(vector);
            index.Add(vector);
        }

        var stored = new float[Dim];
        for (int q = 0; q < 5; q++)
        {
            var query = Vector(20_000 + q);
            var normalizedQuery = (float[])query.Clone();
            Distances.Normalize(normalizedQuery);
            var results = index.Search(query, 10);

            var expected = new List<(int Id, float Score)>();
            for (int id = 0; id < data.Count; id++)
            {
                Assert.True(index.TryGetVector(id, stored));
                expected.Add((id, 1f - Distances.Dot(normalizedQuery, stored)));
            }

            var ordered = expected.OrderBy(x => x.Score).ThenBy(x => x.Id).Take(10).ToList();
            Assert.Equal(10, results.Length);
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(ordered[i].Id, results[i].Id);
                FloatAssertions.Equal(ordered[i].Score, results[i].Score, 0f);
            }
        }
    }

    [Theory]
    [InlineData(64)]
    [InlineData(65)]
    public void Search_Hamming_IsExactAgainstScalarPopcount(int dimensions)
    {
        // Hamming scores are integers: every kernel path agrees exactly, so the exact
        // sequence (including the ascending-id tie-break) must match the naive reference.
        var index = new FlatIndex(new FlatOptions { Dimensions = dimensions, Distance = Distance.Hamming, ParallelThreshold = int.MaxValue });
        var data = new List<float[]>();
        for (int id = 0; id < 300; id++)
        {
            var vector = DeterministicRandom.UniformVector(dimensions, DeterministicRandom.DeriveSeed(0x5eed_8001, id));
            data.Add(vector);
            index.Add(vector);
        }

        for (int q = 0; q < 5; q++)
        {
            var query = DeterministicRandom.UniformVector(dimensions, DeterministicRandom.DeriveSeed(0x5eed_8002, 10_000 + q));
            var results = index.Search(query, 10);

            var packedQuery = BitVectors.PackSignBits(query);
            var expected = data.Select((vector, id) => (Id: id, Score: (float)Distances.HammingScalar(packedQuery, BitVectors.PackSignBits(vector))))
                .OrderBy(x => x.Score)
                .ThenBy(x => x.Id)
                .Take(10)
                .ToList();

            Assert.Equal(10, results.Length);
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(expected[i].Id, results[i].Id);
                FloatAssertions.Equal(expected[i].Score, results[i].Score, 0f);
            }
        }
    }

    // ---- Clustered data: SIMD search must agree with the naive double-precision reference ----

    [Theory]
    [InlineData(Distance.Cosine)]
    [InlineData(Distance.Euclidean)]
    public void Search_ClusteredData_MatchesNaiveScalarReference(Distance distance)
    {
        // 30 clusters of 10 members with far-apart centroids: the top-10 for a query near a
        // centroid is exactly that cluster, with structural margins, so SIMD and naive double
        // rankings must agree on the result set regardless of rounding noise. (DotProduct is
        // covered by the exact-kernel and hand-built tests: raw magnitudes make clustered
        // margins non-structural for it.)
        const int clusters = 30;
        const int members = 10;

        var centroids = new float[clusters][];
        for (int c = 0; c < clusters; c++)
        {
            centroids[c] = DeterministicRandom.GaussianVector(Dim, DeterministicRandom.DeriveSeed(0x5eed_8003, c));
        }

        var index = new FlatIndex(new FlatOptions { Dimensions = Dim, Distance = distance, NormalizeOnAdd = false, ParallelThreshold = int.MaxValue });
        var data = new List<float[]>();
        ulong state = 0x5eed_8004;
        for (int c = 0; c < clusters; c++)
        {
            for (int m = 0; m < members; m++)
            {
                var member = new float[Dim];
                for (int d = 0; d < Dim; d++)
                {
                    member[d] = centroids[c][d] + 0.05f * DeterministicRandom.NextFloat(ref state);
                }

                data.Add(member);
                index.Add(member);
            }
        }

        for (int q = 0; q < clusters; q += 7)
        {
            var query = (float[])centroids[q].Clone();
            var results = index.Search(query, 10);

            var scored = data.Select((vector, id) => (Id: id, Score: NaiveScore(distance, query, vector))).ToList();
            var reference = scored.OrderBy(x => Distances.LowerIsCloser(distance) ? x.Score : -x.Score).ThenBy(x => x.Id).Take(10).ToDictionary(x => x.Id);

            Assert.Equal(10, results.Length);
            foreach (var hit in results)
            {
                Assert.True(reference.ContainsKey(hit.Id), $"id {hit.Id} is not in the naive top-10 (score {hit.Score}).");
                FloatAssertions.Equal(reference[hit.Id].Score, hit.Score, 1e-3f * (1f + MathF.Abs(hit.Score)));
            }

            // Best-first ordering must hold exactly.
            for (int i = 1; i < results.Length; i++)
            {
                bool better = Distances.LowerIsCloser(distance)
                    ? results[i - 1].Score <= results[i].Score
                    : results[i - 1].Score >= results[i].Score;
                Assert.True(better, $"hit {i - 1} ({results[i - 1].Score}) must not be worse than hit {i} ({results[i].Score}).");
            }
        }

        static float NaiveScore(Distance distance, ReadOnlySpan<float> query, ReadOnlySpan<float> vector) => distance switch
        {
            Distance.Cosine => Distances.CosineScalar(query, vector),
            Distance.DotProduct => Distances.DotScalar(query, vector),
            Distance.Euclidean => Distances.EuclideanSquaredScalar(query, vector),
            _ => throw new InvalidOperationException(),
        };
    }

    // ---- Hand-built, known-valid and known-invalid rankings ----

    [Fact]
    public void Euclidean_HandBuiltRanking_ReturnsNearestAndExcludesFarthest()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 2, Distance = Distance.Euclidean, NormalizeOnAdd = false, ParallelThreshold = int.MaxValue });
        index.Add([1f, 0f]);   // id 0: squared L2 to origin = 1
        index.Add([2f, 0f]);   // id 1: = 4
        index.Add([3f, 0f]);   // id 2: = 9
        index.Add([4f, 0f]);   // id 3: = 16 (known-invalid for k=2)

        var results = index.Search([0f, 0f], 2);

        Assert.Equal(2, results.Length);
        Assert.Equal(0, results[0].Id);
        FloatAssertions.Equal(1f, results[0].Score, 0f);
        Assert.Equal(1, results[1].Id);
        FloatAssertions.Equal(4f, results[1].Score, 0f);
        Assert.All(results, hit => Assert.NotEqual(3, hit.Id));
    }

    [Fact]
    public void Cosine_HandBuiltAngles_RankByAngle()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 2, Distance = Distance.Cosine, ParallelThreshold = int.MaxValue });
        index.Add([1f, 0f]);    // id 0: 0° from query
        index.Add([0f, 1f]);    // id 1: 90°
        index.Add([-1f, 0f]);   // id 2: 180°

        var results = index.Search([1f, 0f], 3);

        Assert.Equal(3, results.Length);
        Assert.Equal(0, results[0].Id);
        FloatAssertions.Equal(0f, results[0].Score, 1e-6f);
        Assert.Equal(1, results[1].Id);
        FloatAssertions.Equal(1f, results[1].Score, 1e-6f);
        Assert.Equal(2, results[2].Id);
        FloatAssertions.Equal(2f, results[2].Score, 1e-6f);
    }

    [Fact]
    public void Dot_HandBuiltValues_RankBySimilarity()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 2, Distance = Distance.DotProduct, ParallelThreshold = int.MaxValue });
        index.Add([3f, 3f]);    // id 0: dot with (1,1) = 6
        index.Add([1f, 1f]);    // id 1: = 2
        index.Add([0f, -1f]);   // id 2: = -1

        var results = index.Search([1f, 1f], 2);

        Assert.Equal(2, results.Length);
        Assert.Equal(0, results[0].Id);
        FloatAssertions.Equal(6f, results[0].Score, 0f);
        Assert.Equal(1, results[1].Id);
        FloatAssertions.Equal(2f, results[1].Score, 0f);
        Assert.All(results, hit => Assert.NotEqual(2, hit.Id));
    }

    [Fact]
    public void Hamming_HandBuiltBits_RankByDifferingBits()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 4, Distance = Distance.Hamming, ParallelThreshold = int.MaxValue });
        index.Add([1f, 1f, 1f, 1f]);    // id 0: packed 0000
        index.Add([-1f, 1f, 1f, 1f]);   // id 1: packed 0001 vs query 0000 -> 1
        index.Add([1f, -1f, -1f, 1f]);  // id 2: packed 0101 -> 2

        var results = index.Search([1f, 1f, 1f, 1f], 3);

        Assert.Equal(3, results.Length);
        Assert.Equal(0, results[0].Id);
        FloatAssertions.Equal(0f, results[0].Score, 0f);
        Assert.Equal(1, results[1].Id);
        FloatAssertions.Equal(1f, results[1].Score, 0f);
        Assert.Equal(2, results[2].Id);
        FloatAssertions.Equal(2f, results[2].Score, 0f);
    }

    [Fact]
    public void Ties_IdenticalVectors_BreakByAscendingId()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 8, Distance = Distance.Euclidean, NormalizeOnAdd = false, ParallelThreshold = int.MaxValue });
        var identical = DeterministicRandom.GaussianVector(8, 0x5eed_8005);
        index.Add(identical);
        index.Add(DeterministicRandom.GaussianVector(8, 0x5eed_8006));
        index.Add(identical);
        index.Add(identical);

        var results = index.Search(identical, 3);

        Assert.Equal(3, results.Length);
        Assert.Equal(0, results[0].Id);
        Assert.Equal(2, results[1].Id);
        Assert.Equal(3, results[2].Id);
        FloatAssertions.Equal(results[0].Score, results[1].Score, 0f);
        FloatAssertions.Equal(results[1].Score, results[2].Score, 0f);
        // The distinct random vector is the known-invalid hit for this query.
        Assert.All(results, hit => Assert.NotEqual(1, hit.Id));
    }

    // ---- k semantics and result buffers ----

    [Fact]
    public void Search_EmptyIndex_ReturnsNoHits()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = Dim });

        Assert.Empty(index.Search(Vector(0), 5));

        var buffer = new SearchResult[8];
        Assert.Equal(0, index.Search(Vector(0), 5, buffer));
    }

    [Fact]
    public void Search_KGreaterThanCount_ReturnsAllVectors()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = Dim, ParallelThreshold = int.MaxValue });
        for (int id = 0; id < 7; id++)
        {
            index.Add(Vector(id));
        }

        Assert.Equal(7, index.Search(Vector(999), 50).Length);

        var buffer = new SearchResult[50];
        Assert.Equal(7, index.Search(Vector(999), 50, buffer));
    }

    [Fact]
    public void Search_KEqualsOne_ReturnsSingleBest()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = Dim, ParallelThreshold = int.MaxValue });
        for (int id = 0; id < 100; id++)
        {
            index.Add(Vector(id));
        }

        var results = index.Search(Vector(0), 1);
        Assert.Equal(0, results[0].Id);
        FloatAssertions.Equal(0f, results[0].Score, 1e-4f);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Search_NonPositiveK_Throws(int k)
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = Dim });
        index.Add(Vector(0));

        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(Vector(1), k));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(Vector(1), k, new SearchResult[4]));
    }

    [Fact]
    public void Search_BufferShorterThanK_ReturnsOnlyTheBest()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = Dim, ParallelThreshold = int.MaxValue });
        for (int id = 0; id < 200; id++)
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
            FloatAssertions.Equal(full[i].Score, buffer[i].Score, 0f);
        }
    }

    [Fact]
    public void Search_EmptyBuffer_ReturnsZeroWithoutSearching()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = Dim });
        index.Add(Vector(0));
        Assert.Equal(0, index.Search(Vector(1), 5, Span<SearchResult>.Empty));
    }

    // ---- input validation ----

    [Fact]
    public void Add_WrongDimensions_Throws()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 8 });
        Assert.Throws<ArgumentException>(() => index.Add(new float[7]));
        Assert.Throws<ArgumentException>(() => index.Add(new float[9]));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Add_NonFiniteValues_Throw(float poison)
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 8 });
        var vector = new float[8];
        vector[5] = poison;
        Assert.Throws<ArgumentException>(() => index.Add(vector));
    }

    [Fact]
    public void Search_WrongDimensions_Throws()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 8 });
        index.Add(new float[8]);
        Assert.Throws<ArgumentException>(() => index.Search(new float[7], 3));
        Assert.Throws<ArgumentException>(() => index.Search(new float[9], 3, new SearchResult[3]));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Search_NonFiniteQuery_Throws(float poison)
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 8 });
        index.Add(new float[8]);
        var query = new float[8];
        query[2] = poison;
        Assert.Throws<ArgumentException>(() => index.Search(query, 3));
    }

    // ---- options validation ----

    [Fact]
    public void Options_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new FlatIndex(null!));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Options_NonPositiveDimensions_Throw(int dimensions)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FlatIndex(new FlatOptions { Dimensions = dimensions }));
    }

    [Fact]
    public void Options_NonPositiveInitialCapacity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FlatIndex(new FlatOptions { Dimensions = 8, InitialCapacity = 0 }));
    }

    [Fact]
    public void Options_NegativeParallelThreshold_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FlatIndex(new FlatOptions { Dimensions = 8, ParallelThreshold = -1 }));
    }

    [Fact]
    public void Options_NormalizationOnlyAllowedForCosine()
    {
        Assert.Throws<ArgumentException>(() => new FlatIndex(new FlatOptions { Dimensions = 8, Distance = Distance.DotProduct, NormalizeOnAdd = true }));
        Assert.Throws<ArgumentException>(() => new FlatIndex(new FlatOptions { Dimensions = 8, Distance = Distance.Euclidean, NormalizeOnAdd = true }));
        Assert.Throws<ArgumentException>(() => new FlatIndex(new FlatOptions { Dimensions = 8, Distance = Distance.Hamming, NormalizeOnAdd = true }));

        // Explicit false is fine everywhere, and the default resolves to true only for cosine.
        _ = new FlatIndex(new FlatOptions { Dimensions = 8, Distance = Distance.DotProduct, NormalizeOnAdd = false });
        _ = new FlatIndex(new FlatOptions { Dimensions = 8, Distance = Distance.Euclidean, NormalizeOnAdd = false });
        _ = new FlatIndex(new FlatOptions { Dimensions = 8, Distance = Distance.Hamming, NormalizeOnAdd = false });
    }

    // ---- storage behaviour ----

    [Fact]
    public void Add_ReturnsSequentialIds_AndGrowsBeyondInitialCapacity()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = Dim, InitialCapacity = 2, ParallelThreshold = int.MaxValue });
        var stored = new float[Dim];

        for (int id = 0; id < 200; id++)
        {
            var vector = Vector(id, seed: 0x5eed_8007);
            Assert.Equal(id, index.Add(vector));
        }

        Assert.Equal(200, index.Count);
        for (int id = 0; id < 200; id++)
        {
            Assert.True(index.TryGetVector(id, stored));
            Assert.Equal(Vector(id, seed: 0x5eed_8007), stored);
        }
    }

    [Fact]
    public void TryGetVector_CosineDefault_NormalizesStorage()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 4 });
        index.Add([3f, 0f, 0f, 0f]);

        var stored = new float[4];
        Assert.True(index.TryGetVector(0, stored));
        FloatAssertions.Equal(1f, Distances.Norm(stored), 1e-6f);
        Assert.Equal([1f, 0f, 0f, 0f], stored);
    }

    [Fact]
    public void TryGetVector_NormalizeExplicitlyOff_StoresVerbatim()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 4, NormalizeOnAdd = false });
        index.Add([3f, 0f, 0f, 0f]);

        var stored = new float[4];
        Assert.True(index.TryGetVector(0, stored));
        Assert.Equal([3f, 0f, 0f, 0f], stored);
    }

    [Fact]
    public void TryGetVector_Hamming_ReconstructsSigns()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 3, Distance = Distance.Hamming });
        index.Add([-1f, 2f, -3f]);

        var stored = new float[3];
        Assert.True(index.TryGetVector(0, stored));
        Assert.Equal([-1f, 1f, -1f], stored);
    }

    [Fact]
    public void TryGetVector_OutOfRangeOrInvalidDestination_ThrowsOrReturnsFalse()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 4 });
        index.Add(new float[4]);

        Assert.False(index.TryGetVector(1, new float[4]));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.TryGetVector(-1, new float[4]));
        Assert.Throws<ArgumentException>(() => index.TryGetVector(0, new float[3]));
    }

    [Fact]
    public void Index_ExposesConfiguration()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 32, Distance = Distance.DotProduct, NormalizeOnAdd = false });
        Assert.Equal(32, index.Dimensions);
        Assert.Equal(Distance.DotProduct, index.Distance);
        Assert.Equal(0, index.Count);
    }
}
