using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Flat;

public class FlatParallelTests
{
    public static TheoryData<int> Sizes { get; } = new() { 2, 3, 64, 1000, 4096, 5000 };

    private static FlatIndex Build(int count, Distance distance, int parallelThreshold, int dimensions = 64, List<float[]>? data = null)
    {
        var index = new FlatIndex(new FlatOptions
        {
            Dimensions = dimensions,
            Distance = distance,
            NormalizeOnAdd = distance == Distance.DotProduct ? false : null,
            ParallelThreshold = parallelThreshold,
        });

        for (int id = 0; id < count; id++)
        {
            var vector = DeterministicRandom.GaussianVector(dimensions, DeterministicRandom.DeriveSeed(0x5eed_9000, id));
            data?.Add(vector);
            index.Add(vector);
        }

        return index;
    }

    [Theory]
    [MemberData(nameof(Sizes))]
    public void ParallelSearch_IsIdenticalToSequential(int count)
    {
        var sequential = Build(count, Distance.Cosine, int.MaxValue);
        var parallel = Build(count, Distance.Cosine, 1);

        for (int q = 0; q < 5; q++)
        {
            var query = DeterministicRandom.GaussianVector(64, DeterministicRandom.DeriveSeed(0x5eed_9001, 50_000 + q));

            foreach (int k in (int[])[1, 5, 33])
            {
                var expected = sequential.Search(query, k);
                var actual = parallel.Search(query, k);

                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                {
                    Assert.Equal(expected[i].Id, actual[i].Id);
                    FloatAssertions.Equal(expected[i].Score, actual[i].Score, 0f);
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Sizes))]
    public void ParallelSearch_IsDeterministicAcrossRepeats(int count)
    {
        var parallel = Build(count, Distance.Cosine, 1);
        var query = DeterministicRandom.GaussianVector(64, 0x5eed_9002);

        var first = parallel.Search(query, 7);
        for (int repeat = 0; repeat < 3; repeat++)
        {
            var again = parallel.Search(query, 7);
            Assert.Equal(first, again);
        }
    }

    [Theory]
    [InlineData(Distance.DotProduct)]
    [InlineData(Distance.Euclidean)]
    [InlineData(Distance.Hamming)]
    public void ParallelSearch_AllMetrics_IdenticalToSequential(Distance distance)
    {
        int dimensions = distance == Distance.Hamming ? 65 : 64; // partial last word for Hamming
        var sequential = Build(3000, distance, int.MaxValue, dimensions);
        var parallel = Build(3000, distance, 1, dimensions);

        for (int q = 0; q < 5; q++)
        {
            var query = DeterministicRandom.UniformVector(dimensions, DeterministicRandom.DeriveSeed(0x5eed_9003, 60_000 + q));
            var expected = sequential.Search(query, 12);
            var actual = parallel.Search(query, 12);

            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void ParallelSearch_DefaultThreshold_EngagesForLargeCollections()
    {
        // Default ParallelThreshold is 4096: a 5000-vector collection searches in parallel and
        // must still agree exactly with a forced-sequential index over the same data.
        var sequential = Build(5000, Distance.Cosine, int.MaxValue);
        var parallel = Build(5000, Distance.Cosine, 4096);

        var query = DeterministicRandom.GaussianVector(64, 0x5eed_9004);
        Assert.Equal(sequential.Search(query, 10), parallel.Search(query, 10));
    }

    [Fact]
    public void ParallelSearch_ThresholdZero_SingleVector_StaysCorrect()
    {
        var index = Build(1, Distance.Cosine, 0);
        var query = DeterministicRandom.GaussianVector(64, 0x5eed_9005);

        var results = index.Search(query, 5);
        Assert.Single(results);
        Assert.Equal(0, results[0].Id);
    }

    [Fact]
    public void ParallelSearch_SpanOverload_MatchesArrayOverload()
    {
        var parallel = Build(2048, Distance.Euclidean, 1);
        var query = DeterministicRandom.GaussianVector(64, 0x5eed_9006);

        var expected = parallel.Search(query, 10);
        var buffer = new SearchResult[10];
        int written = parallel.Search(query, 10, buffer);

        Assert.Equal(10, written);
        Assert.Equal(expected, buffer);
    }
}
