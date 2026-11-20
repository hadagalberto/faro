using System.Diagnostics;
using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.RecallTests;

/// <summary>
/// Recall gates for the HNSW index. Flat (the exact index from milestone 1) is the ground truth:
/// recall@10 is the fraction of the exact top-10 that HNSW returns at efSearch=64.
/// </summary>
public class HnswRecallTests
{
    private const int Queries = 25;
    private const int K = 10;
    private const int EfSearch = 64;

    private readonly ITestOutputHelper _output;

    public HnswRecallTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Hnsw_Clustered50k128d_RecallAt10_AtLeast095()
    {
        const int count = 50_000;
        const int dimensions = 128;
        const int clusters = 500;
        const int members = count / clusters;

        var (data, queries) = BuildClusteredData(count, dimensions, clusters, members);

        var flat = new FlatIndex(new FlatOptions { Dimensions = dimensions, Distance = Distance.Cosine });
        var buildWatch = Stopwatch.StartNew();
        foreach (var vector in data)
        {
            flat.Add(vector);
        }

        var hnsw = new HnswIndex(new HnswOptions
        {
            Dimensions = dimensions,
            Distance = Distance.Cosine,
            M = 16,
            EfConstruction = 200,
            EfSearch = EfSearch,
            Seed = 0x5eed_1234_5678,
        });
        foreach (var vector in data)
        {
            hnsw.Add(vector);
        }
        buildWatch.Stop();

        _output.WriteLine($"built {count:N0} x {dimensions} vectors: flat + hnsw in {buildWatch.Elapsed.TotalSeconds:F1}s (hnsw max level {hnsw.MaxLevel})");
        _output.WriteLine("| dataset                     | N     | d   | ef  | recall@10 | us/query |");
        _output.WriteLine("|-----------------------------|-------|-----|-----|-----------|----------|");

        double sumRecall = 0;
        double minRecall = 1;
        double totalLookup = 0;
        int strongQueries = 0;

        foreach (var query in queries)
        {
            var truth = flat.Search(query, K).Select(hit => hit.Id).ToHashSet();
            var watch = Stopwatch.StartNew();
            var hits = hnsw.Search(query, K, efSearch: EfSearch);
            watch.Stop();

            double recall = hits.Count(hit => truth.Contains(hit.Id)) / (double)K;
            sumRecall += recall;
            minRecall = Math.Min(minRecall, recall);
            totalLookup += watch.Elapsed.TotalMilliseconds * 1000;
            if (recall >= 0.9)
            {
                strongQueries++;
            }
        }

        double meanRecall = sumRecall / Queries;
        double usPerQuery = totalLookup / Queries;
        _output.WriteLine($"| clustered-gaussian-50k-128d | {count,5} | {dimensions,3} | {EfSearch,3} | {meanRecall,9:F3} | {usPerQuery,8:F1} |");
        _output.WriteLine($"mean recall@10 = {meanRecall:F3}, min = {minRecall:F3}, queries with recall >= 0.9 = {strongQueries}/{Queries}");

        // recall@k is conventionally the mean over the query set. The second gate keeps the
        // per-query tail honest: no more than one in ten queries may fall below 0.9.
        Assert.True(
            meanRecall >= 0.95,
            $"HNSW mean recall@10 at efSearch={EfSearch} was {meanRecall:F3}, below the 0.95 gate.");
        Assert.True(
            strongQueries >= Queries * 0.9,
            $"only {strongQueries}/{Queries} queries reached recall@10 >= 0.9.");
    }

    [Fact]
    public void Hnsw_RecallImprovesWithEfSearch()
    {
        const int count = 8_000;
        const int dimensions = 64;
        const int clusters = 160;
        const int members = count / clusters;

        var (data, queries) = BuildClusteredData(count, dimensions, clusters, members);
        var flat = new FlatIndex(new FlatOptions { Dimensions = dimensions, Distance = Distance.Cosine });
        var hnsw = new HnswIndex(new HnswOptions
        {
            Dimensions = dimensions,
            Distance = Distance.Cosine,
            M = 12,
            EfConstruction = 120,
            EfSearch = 16,
            Seed = 0x5eed_4321,
        });

        foreach (var vector in data)
        {
            flat.Add(vector);
            hnsw.Add(vector);
        }

        var truths = queries.Select(q => flat.Search(q, K).Select(h => h.Id).ToHashSet()).ToArray();

        var low = MeasureRecall(hnsw, queries, truths, ef: 16);
        var high = MeasureRecall(hnsw, queries, truths, ef: 128);

        _output.WriteLine($"| ef=16  | 8k x 64 clustered | recall@10 = {low:F3} |");
        _output.WriteLine($"| ef=128 | 8k x 64 clustered | recall@10 = {high:F3} |");

        Assert.True(high >= low - 1e-9, $"larger ef must not reduce mean recall ({low:F3} -> {high:F3})");
        Assert.True(high >= 0.95, $"recall at ef=128 was {high:F3}");
    }

    private static double MeasureRecall(HnswIndex index, List<float[]> queries, HashSet<int>[] truths, int ef)
    {
        double sum = 0;
        for (int i = 0; i < queries.Count; i++)
        {
            var hits = index.Search(queries[i], K, efSearch: ef);
            sum += hits.Count(hit => truths[i].Contains(hit.Id)) / (double)K;
        }

        return sum / queries.Count;
    }

    private static (List<float[]> Data, List<float[]> Queries) BuildClusteredData(
        int count, int dimensions, int clusters, int members)
    {
        var centroids = new float[clusters][];
        for (int c = 0; c < clusters; c++)
        {
            centroids[c] = DeterministicRandom.GaussianVector(
                dimensions, DeterministicRandom.DeriveSeed(0x5eed_c100, c));
        }

        var data = new List<float[]>(count);
        ulong noise = 0x5eed_c101;
        for (int c = 0; c < clusters; c++)
        {
            for (int m = 0; m < members && data.Count < count; m++)
            {
                var point = new float[dimensions];
                for (int d = 0; d < dimensions; d++)
                {
                    point[d] = centroids[c][d] + 0.05f * DeterministicRandom.NextFloat(ref noise);
                }

                data.Add(point);
            }
        }

        var queries = new List<float[]>(Queries);
        ulong queryNoise = 0x5eed_c102;
        for (int q = 0; q < Queries; q++)
        {
            int cluster = (int)((long)q * clusters / Queries);
            var query = new float[dimensions];
            for (int d = 0; d < dimensions; d++)
            {
                query[d] = centroids[cluster][d] + 0.02f * DeterministicRandom.NextFloat(ref queryNoise);
            }

            queries.Add(query);
        }

        return (data, queries);
    }
}
