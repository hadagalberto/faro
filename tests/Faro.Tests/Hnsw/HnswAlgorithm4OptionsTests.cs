using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Hnsw;

/// <summary>
/// The Algorithm 4 toggles must actually change the graph they build, not just exist as options.
/// All builds use the same data and seed, so any difference is caused by the option under test.
/// </summary>
public class HnswAlgorithm4OptionsTests
{
    private const int Count = 400;
    private const int Dim = 32;
    private const int M = 8;

    [Fact]
    public void ExtendCandidates_ChangesNeighbourSelection()
    {
        var data = Data();
        int[] extended = Signature(Build(data, extend: true, keepPruned: false, mc: 1.0));
        int[] plain = Signature(Build(data, extend: false, keepPruned: false, mc: 1.0));

        Assert.NotEqual(extended, plain);
        AssertAllDegreesBounded(extended);
        AssertAllDegreesBounded(plain);
    }

    [Fact]
    public void KeepPrunedConnections_ChangesNeighbourSelection()
    {
        var data = Data();
        int[] kept = Signature(Build(data, extend: true, keepPruned: true, mc: 1.0));
        int[] pruned = Signature(Build(data, extend: true, keepPruned: false, mc: 1.0));

        Assert.NotEqual(kept, pruned);
        AssertAllDegreesBounded(kept);
        AssertAllDegreesBounded(pruned);
    }

    [Fact]
    public void Mc_ChangesDiversityTest()
    {
        var data = Data();
        int[] strict = Signature(Build(data, extend: true, keepPruned: false, mc: 1.0));
        int[] relaxed = Signature(Build(data, extend: true, keepPruned: false, mc: 2.0));

        Assert.NotEqual(strict, relaxed);
        AssertAllDegreesBounded(strict);
        AssertAllDegreesBounded(relaxed);
    }

    [Theory]
    [InlineData(true, false, 1.0)]
    [InlineData(false, false, 1.0)]
    [InlineData(true, true, 1.0)]
    [InlineData(true, false, 2.0)]
    public void EveryToggleCombination_SearchStillMatchesFlatWithWideBeam(
        bool extend, bool keepPruned, double mc)
    {
        var data = Data();
        var hnsw = Build(data, extend, keepPruned, mc);
        var flat = new FlatIndex(new FlatOptions
        {
            Dimensions = Dim,
            Distance = Distance.Cosine,
            ParallelThreshold = int.MaxValue,
        });
        foreach (float[] vector in data)
        {
            flat.Add(vector);
        }

        for (int q = 0; q < 5; q++)
        {
            float[] query = DeterministicRandom.UnitVector(Dim, DeterministicRandom.DeriveSeed(0x5eed_f002, q));
            SearchResult[] expected = flat.Search(query, 10);
            SearchResult[] actual = hnsw.Search(query, 10, efSearch: Count);
            Assert.Equal(expected.Select(hit => hit.Id), actual.Select(hit => hit.Id));
        }
    }

    private static HnswIndex Build(float[][] data, bool extend, bool keepPruned, double mc)
    {
        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = Dim,
            Distance = Distance.Cosine,
            M = M,
            EfConstruction = 120,
            EfSearch = 64,
            Seed = 0x5eed_f001,
            ExtendCandidates = extend,
            KeepPrunedConnections = keepPruned,
            Mc = mc,
        });
        foreach (float[] vector in data)
        {
            index.Add(vector);
        }

        return index;
    }

    private static float[][] Data()
    {
        var data = new float[Count][];
        for (int i = 0; i < Count; i++)
        {
            data[i] = DeterministicRandom.UnitVector(Dim, DeterministicRandom.DeriveSeed(0x5eed_f000, i));
        }

        return data;
    }

    /// <summary>Deterministic sorted (node, layer-0 neighbour) pair signature of the graph.</summary>
    private static int[] Signature(HnswIndex index)
    {
        var pairs = new List<int>();
        var view = index.CurrentView;
        for (int id = 0; id < view.Count; id++)
        {
            foreach (int neighbour in view.Nodes[id]!.GetLinks(0))
            {
                pairs.Add(id);
                pairs.Add(neighbour);
            }
        }

        return pairs.ToArray();
    }

    private static void AssertAllDegreesBounded(int[] signature)
    {
        var counts = new Dictionary<int, int>();
        for (int i = 0; i < signature.Length; i += 2)
        {
            counts[signature[i]] = counts.GetValueOrDefault(signature[i]) + 1;
        }

        Assert.All(counts.Values, degree => Assert.InRange(degree, 0, 2 * M));
    }
}
