using Faro;
using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Hnsw;

/// <summary>
/// Structural invariants of the HNSW graph: degree bounds, layer reachability, tombstone
/// behaviour and the guarantee that compaction leaves no node referencing a deleted one.
/// </summary>
public class HnswInvariantTests
{
    private const int Dim = 24;

    private static HnswIndex Build(int count, int m = 16, int efConstruction = 120, ulong seed = 0x5eed_f000)
    {
        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = Dim,
            Distance = Distance.Euclidean,
            NormalizeOnAdd = false,
            M = m,
            EfConstruction = efConstruction,
            EfSearch = 64,
            Seed = unchecked((long)seed),
        });

        for (int id = 0; id < count; id++)
        {
            index.Add(DeterministicRandom.GaussianVector(Dim, DeterministicRandom.DeriveSeed(seed, id)));
        }

        return index;
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void DegreeBounds_NeverExceedM0AtLayerZero_AndMAbove(int m)
    {
        var index = Build(count: 400, m: m);
        var view = index.CurrentView;

        for (int id = 0; id < view.Count; id++)
        {
            HnswNode node = view.Nodes[id]!;
            for (int layer = 0; layer <= node.Level; layer++)
            {
                int[] links = node.GetLinks(layer);
                int bound = layer == 0 ? 2 * m : m;
                Assert.True(links.Length <= bound, $"node {id} layer {layer} has {links.Length} links, bound {bound}.");
                Assert.DoesNotContain(id, links); // no self-links
                Assert.Equal(links.Length, links.Distinct().Count()); // no duplicates
            }
        }
    }

    [Fact]
    public void LayerZero_AllLiveNodesReachableFromEntryPoint()
    {
        var index = Build(count: 600, m: 6);
        AssertReachable(index, layer: 0, expectedCount: index.LiveCount);
    }

    [Fact]
    public void EveryUpperLayer_ReachableFromEntryPoint()
    {
        var index = Build(count: 600, m: 6);
        var view = index.CurrentView;
        int maxLevel = view.MaxLevel;
        Assert.True(maxLevel >= 1, "expected a graph with upper layers");

        for (int layer = 1; layer <= maxLevel; layer++)
        {
            int expected = 0;
            for (int id = 0; id < view.Count; id++)
            {
                if (view.Nodes[id]!.Level >= layer)
                {
                    expected++;
                }
            }

            AssertReachable(index, layer, expected);
        }
    }

    [Fact]
    public void LinksAreMostlySymmetric()
    {
        // HNSW prunes asymmetrically, but the vast majority of layer-0 links are bidirectional.
        var index = Build(count: 400, m: 8);
        var view = index.CurrentView;
        long directed = 0;
        long reciprocal = 0;

        for (int id = 0; id < view.Count; id++)
        {
            HnswNode node = view.Nodes[id]!;
            foreach (int other in node.GetLinks(0))
            {
                directed++;
                if (Array.IndexOf(view.Nodes[other]!.GetLinks(0), id) >= 0)
                {
                    reciprocal++;
                }
            }
        }

        Assert.True(directed > 0);
        Assert.True(reciprocal >= directed / 2, $"only {reciprocal}/{directed} links are reciprocal");
    }

    [Fact]
    public void Delete_RemovesLinksToTombstone_AndSearchNeverReturnsIt()
    {
        var index = Build(count: 300, m: 6);
        var victims = new[] { 5, 42, 137, 250 };
        foreach (int victim in victims)
        {
            Assert.True(index.Delete(victim));
            Assert.False(index.Delete(victim)); // idempotent
        }

        Assert.Equal(300, index.Count);
        Assert.Equal(296, index.LiveCount);

        var view = index.CurrentView;
        Assert.All(victims, id => Assert.True(view.Nodes[id]!.Deleted));
        for (int id = 0; id < view.Count; id++)
        {
            HnswNode node = view.Nodes[id]!;
            for (int layer = 0; layer <= node.Level; layer++)
            {
                foreach (int link in node.GetLinks(layer))
                {
                    Assert.False(victims.Contains(link), $"node {id} layer {layer} still links to deleted node {link} (owner deleted: {node.Deleted})");
                }
            }
        }

        for (int q = 0; q < 10; q++)
        {
            var query = DeterministicRandom.GaussianVector(Dim, DeterministicRandom.DeriveSeed(0x5eed_f001, 10_000 + q));
            var hits = index.Search(query, 20, efSearch: 200);
            Assert.All(hits, hit => Assert.DoesNotContain(hit.Id, victims));
        }
    }

    [Fact]
    public void Delete_KeepsEveryLiveNodeReachable()
    {
        var index = Build(count: 500, m: 8);
        var rngSeeds = DeterministicRandom.DeriveSeed(0x5eed_f002, 0);
        ulong state = rngSeeds;
        for (int i = 0; i < 100; i++)
        {
            int victim = DeterministicRandom.NextInt(ref state, index.Count);
            index.Delete(victim);
        }

        AssertReachable(index, layer: 0, expectedCount: index.LiveCount);
    }

    [Fact]
    public void Compact_RemovesTombstones_NoLinksPointToThem_LiveDataPreserved()
    {
        var index = Build(count: 400, m: 6);
        var removed = new HashSet<int>();
        ulong state = 0x5eed_f003;
        for (int i = 0; i < 60; i++)
        {
            int victim = DeterministicRandom.NextInt(ref state, index.Count);
            if (removed.Add(victim))
            {
                Assert.True(index.Delete(victim));
            }
        }

        var survivors = Enumerable.Range(0, index.Count).Except(removed).ToArray();
        var before = new Dictionary<int, float[]>();
        var scratch = new float[Dim];
        foreach (int id in survivors)
        {
            var vector = new float[Dim];
            Assert.True(index.TryGetVector(id, vector));
            before[id] = vector;
        }

        index.Compact();

        Assert.Equal(400, index.Count); // ids are stable; no reuse
        Assert.Equal(survivors.Length, index.LiveCount);

        var view = index.CurrentView;
        foreach (int id in removed)
        {
            Assert.Null(view.Nodes[id]);
        }

        for (int id = 0; id < view.Count; id++)
        {
            HnswNode? node = view.Nodes[id];
            if (node is null)
            {
                continue;
            }

            Assert.False(node.Deleted);
            for (int layer = 0; layer <= node.Level; layer++)
            {
                foreach (int link in node.GetLinks(layer))
                {
                    Assert.False(removed.Contains(link), $"live node {id} still links to deleted node {link}");
                    Assert.NotNull(view.Nodes[link]);
                }
            }

            Assert.True(index.TryGetVector(id, scratch));
            Assert.Equal(before[id], scratch);
        }

        AssertReachable(index, layer: 0, expectedCount: survivors.Length);
    }

    [Fact]
    public void Compact_IsIdempotent()
    {
        var index = Build(count: 200, m: 6);
        index.Delete(10);
        index.Delete(20);
        index.Compact();

        var first = Snapshot(index, 200);
        index.Compact();
        var second = Snapshot(index, 200);

        Assert.Equal(first, second);
    }

    [Fact]
    public void Compact_OnIndexWithoutDeletes_PreservesSearchResults()
    {
        var index = Build(count: 250, m: 6);
        var query = DeterministicRandom.GaussianVector(Dim, 0x5eed_f004);
        var before = index.Search(query, 10, efSearch: 200);

        index.Compact();
        var after = index.Search(query, 10, efSearch: 200);

        Assert.Equal(before, after);
    }

    // ---- helpers ----

    private static void AssertReachable(HnswIndex index, int layer, int expectedCount)
    {
        var view = index.CurrentView;
        Assert.True(view.EntryPointId >= 0, "entry point must exist");

        var visited = new bool[view.Count];
        var queue = new Queue<int>();
        queue.Enqueue(view.EntryPointId);
        visited[view.EntryPointId] = true;
        int seen = 0;

        while (queue.Count > 0)
        {
            int id = queue.Dequeue();
            HnswNode? node = view.Nodes[id];
            if (node is null)
            {
                continue;
            }

            // The entry point may be tombstoned; its links are still valid traversal edges.
            if (!node.Deleted)
            {
                seen++;
            }

            if (layer > node.Level)
            {
                continue;
            }

            for (int source = 0; source < 2; source++)
            {
                int[] adjacent = source == 0 ? node.GetLinks(layer) : node.GetBackbone(layer);
                foreach (int next in adjacent)
                {
                    if ((uint)next >= (uint)view.Count)
                    {
                        continue;
                    }

                    bool traversable = next == view.EntryPointId || view.Nodes[next] is { Deleted: false };
                    if (!visited[next] && traversable)
                    {
                        visited[next] = true;
                        queue.Enqueue(next);
                    }
                }
            }
        }

        if (seen != expectedCount)
        {
            var missing = Enumerable.Range(0, view.Count)
                .Where(id => !visited[id] && view.Nodes[id] is { Deleted: false })
                .Take(10)
                .ToArray();
            var details = string.Join(" | ", missing.Select(id =>
            {
                var node = view.Nodes[id]!;
                int incoming = Enumerable.Range(0, view.Count)
                    .Where(other => view.Nodes[other] is { } n && layer <= n.Level
                        && (Array.IndexOf(n.GetLinks(layer), id) >= 0 || Array.IndexOf(n.GetBackbone(layer), id) >= 0))
                    .Count();
                return $"{id}: lvl={node.Level}, out=[{string.Join(",", node.GetLinks(layer))}], spine=[{string.Join(",", node.GetBackbone(layer))}], in={incoming}";
            }));
            Assert.Fail($"reachable {seen}/{expectedCount} at layer {layer}; entry {view.EntryPointId}; missing: {details}");
        }
    }

    private static List<(int Level, string Links)> Snapshot(HnswIndex index, int count)
    {
        var view = index.CurrentView;
        var result = new List<(int, string)>();
        for (int id = 0; id < count; id++)
        {
            HnswNode? node = view.Nodes[id];
            result.Add(node is null
                ? (-1, string.Empty)
                : (node.Level, string.Join(";", node.Neighbors.Select(layer => string.Join(",", layer)))));
        }

        return result;
    }
}
