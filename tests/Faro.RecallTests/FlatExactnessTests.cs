using System.Diagnostics;
using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.RecallTests;

/// <summary>
/// Quality gates for the exact Flat index. Flat is the ground-truth engine every approximate
/// index will be compared against in later milestones, so its recall against a naive
/// double-precision brute force must be exactly 1.0 and its parallel path must be identical.
/// </summary>
public class FlatExactnessTests
{
    private const int Queries = 25;
    private const int K = 10;
    private readonly ITestOutputHelper _output;

    public FlatExactnessTests(ITestOutputHelper output)
    {
        _output = output;
    }

    public static TheoryData<string, Distance, int, int, int> ClusteredCases { get; } = new()
    {
        // (name, metric, dimensions, clusters, clusterSize == K members)
        { "cosine-128d", Distance.Cosine, 128, 1000, K },
        { "euclidean-128d", Distance.Euclidean, 128, 1000, K },
        { "cosine-768d", Distance.Cosine, 768, 500, K },
    };

    [Theory]
    [MemberData(nameof(ClusteredCases))]
    public void Flat_RecallAtK_AgainstNaiveBruteForce_IsExactlyOne(string name, Distance distance, int dimensions, int clusters, int members)
    {
        // Clustered Gaussian data (SPEC recall-suite style): each query sits near a centroid
        // whose cluster has exactly K members, so the top-K is structurally separated from the
        // rest of the corpus by margins orders of magnitude above float rounding. SIMD and
        // naive double rankings must therefore agree exactly, query after query.
        var (index, queries, data) = BuildClustered(distance, dimensions, clusters, members);

        double minRecall = 1.0;
        foreach (var query in queries)
        {
            var hits = index.Search(query, K);
            var reference = NaiveTopK(distance, data, query, K);

            var referenceIds = reference.Select(r => r.Id).ToHashSet();
            double recall = hits.Count(h => referenceIds.Contains(h.Id)) / (double)K;
            minRecall = Math.Min(minRecall, recall);

            // Scores must match the naive double reference within float tolerance.
            foreach (var hit in hits)
            {
                var expected = reference.First(r => r.Id == hit.Id);
                Assert.True(
                    MathF.Abs(hit.Score - expected.Score) <= 1e-3f * (1f + MathF.Abs(expected.Score)),
                    $"{name}: score for id {hit.Id} was {hit.Score}, naive reference {expected.Score}.");
            }
        }

        // Timing pass: search only.
        var searchWatch = Stopwatch.StartNew();
        foreach (var query in queries)
        {
            index.Search(query, K);
        }

        searchWatch.Stop();
        double microsecondsPerSearch = searchWatch.Elapsed.TotalMilliseconds * 1000 / Queries;

        Assert.Equal(1.0, minRecall, 6);
        double vectorsPerSecond = clusters * members / (searchWatch.Elapsed.TotalSeconds / Queries);
        _output.WriteLine($"| {name,-18} | {clusters * members,6} | {dimensions,4} | {K,2} | recall@10 = {minRecall:F3} | {microsecondsPerSearch,9:F1} us/query | {vectorsPerSecond / 1_000_000:F2} M vectors/s |");
    }

    [Theory]
    [MemberData(nameof(ClusteredCases))]
    public void Flat_ParallelSearch_IsIdenticalToSequential_AndEquallyExact(string name, Distance distance, int dimensions, int clusters, int members)
    {
        var (sequential, queries, data) = BuildClustered(distance, dimensions, clusters, members);
        var parallel = RebuildWithThreshold(sequential, data, 1);

        foreach (var query in queries)
        {
            var expected = sequential.Search(query, K);
            var actual = parallel.Search(query, K);

            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i].Id, actual[i].Id);
                FloatAssertions.Equal(expected[i].Score, actual[i].Score, 0f);
            }
        }

        var parallelWatch = Stopwatch.StartNew();
        foreach (var query in queries)
        {
            parallel.Search(query, K);
        }

        parallelWatch.Stop();
        double microsecondsPerSearch = parallelWatch.Elapsed.TotalMilliseconds * 1000 / Queries;
        _output.WriteLine($"| {name,-18} | parallel identical to sequential | {microsecondsPerSearch,9:F1} us/query |");
    }

    [Fact]
    public void Flat_Hamming_UniformData_MatchesNaivePopcountExactly()
    {
        // Hamming scores are exact integers, so the Flat result sequence (ties included) must
        // be bit-identical to a naive popcount brute force on uniform random data.
        const int n = 5_000;
        const int dim = 768;

        var index = new FlatIndex(new FlatOptions { Dimensions = dim, Distance = Distance.Hamming, ParallelThreshold = 4096 });
        var data = new List<float[]>(n);
        for (int id = 0; id < n; id++)
        {
            var vector = DeterministicRandom.UniformVector(dim, DeterministicRandom.DeriveSeed(0x5eed_c000, id));
            data.Add(vector);
            index.Add(vector);
        }

        var watch = Stopwatch.StartNew();
        for (int q = 0; q < Queries; q++)
        {
            var query = DeterministicRandom.UniformVector(dim, DeterministicRandom.DeriveSeed(0x5eed_c001, 100_000 + q));
            var hits = index.Search(query, K);
            var reference = NaiveTopK(Distance.Hamming, data, query, K);

            Assert.Equal(reference.Count, hits.Length);
            for (int i = 0; i < hits.Length; i++)
            {
                Assert.Equal(reference[i].Id, hits[i].Id);
                FloatAssertions.Equal(reference[i].Score, hits[i].Score, 0f);
            }
        }

        watch.Stop();
        double microsecondsPerSearch = watch.Elapsed.TotalMilliseconds * 1000 / Queries;
        _output.WriteLine($"| hamming-768d       | {n,6} | {dim,4} | {K,2} | recall@10 = 1.000 | {microsecondsPerSearch,9:F1} us/query | (integer-exact) |");
    }

    private static (FlatIndex Index, List<float[]> Queries, List<float[]> Data) BuildClustered(
        Distance distance, int dimensions, int clusters, int members)
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = dimensions, Distance = distance, NormalizeOnAdd = false });
        var data = new List<float[]>(clusters * members);

        for (int c = 0; c < clusters; c++)
        {
            var centroid = DeterministicRandom.GaussianVector(dimensions, DeterministicRandom.DeriveSeed(0x5eed_c002, c));
            for (int m = 0; m < members; m++)
            {
                var member = new float[dimensions];
                ulong noise = DeterministicRandom.DeriveSeed(0x5eed_c003, c * 1000 + m);
                for (int d = 0; d < dimensions; d++)
                {
                    member[d] = centroid[d] + 0.05f * DeterministicRandom.NextFloat(ref noise);
                }

                data.Add(member);
                index.Add(member);
            }
        }

        var queries = new List<float[]>(Queries);
        for (int q = 0; q < Queries; q++)
        {
            int cluster = (int)((long)q * clusters / Queries);
            var centroid = DeterministicRandom.GaussianVector(dimensions, DeterministicRandom.DeriveSeed(0x5eed_c002, cluster));
            var query = new float[dimensions];
            ulong noise = DeterministicRandom.DeriveSeed(0x5eed_c004, q);
            for (int d = 0; d < dimensions; d++)
            {
                query[d] = centroid[d] + 0.02f * DeterministicRandom.NextFloat(ref noise);
            }

            queries.Add(query);
        }

        return (index, queries, data);
    }

    private static FlatIndex RebuildWithThreshold(FlatIndex source, List<float[]> data, int parallelThreshold)
    {
        var index = new FlatIndex(new FlatOptions
        {
            Dimensions = source.Dimensions,
            Distance = source.Distance,
            NormalizeOnAdd = false,
            ParallelThreshold = parallelThreshold,
        });

        foreach (var vector in data)
        {
            index.Add(vector);
        }

        return index;
    }

    private static List<(int Id, float Score)> NaiveTopK(Distance distance, List<float[]> data, ReadOnlySpan<float> query, int k)
    {
        var scored = new List<(int Id, float Score)>(data.Count);
        if (distance == Distance.Hamming)
        {
            var packedQuery = BitVectors.PackSignBits(query);
            for (int id = 0; id < data.Count; id++)
            {
                scored.Add((id, Distances.Hamming(packedQuery, BitVectors.PackSignBits(data[id]))));
            }
        }
        else
        {
            for (int id = 0; id < data.Count; id++)
            {
                float score = distance switch
                {
                    Distance.Cosine => Distances.CosineScalar(query, data[id]),
                    Distance.Euclidean => Distances.EuclideanSquaredScalar(query, data[id]),
                    _ => throw new InvalidOperationException(),
                };
                scored.Add((id, score));
            }
        }

        return scored
            .OrderBy(x => x.Score)
            .ThenBy(x => x.Id)
            .Take(k)
            .ToList();
    }
}
