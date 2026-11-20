using System.Diagnostics;
using Faro.Persistence;
using Faro.Testing;
using Faro.Tests.Persistence;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Hnsw;

/// <summary>
/// Restore used to call Compact(), rebuilding the whole graph from vectors on every open
/// (82.6 s for 20k x 128 versus 0.14 s to write the checkpoint). Stored adjacency pages must be
/// loaded directly. Replace also has to relink nodes after a vector moves.
/// </summary>
public class HnswRestoreTests : PersistenceTestBase
{
    [Fact]
    public void Reopen_LoadsStoredGraphWithoutRebuilding()
    {
        const int Dimensions = 32;
        const int Records = 2000;
        string path = DbPath();

        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options(Dimensions));
            for (int i = 0; i < Records; i++)
            {
                docs.Upsert(i, DeterministicRandom.UniformVector(Dimensions, DeterministicRandom.DeriveSeed(7, i)), MakeDoc(i));
            }

            db.Checkpoint();
        }

        var clock = Stopwatch.StartNew();
        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> loaded = reopened.GetOrCreateCollection("docs", Options(Dimensions));
        clock.Stop();

        // Hand-derived contract: reopen is a page load, not a graph rebuild, so it must be far
        // below the multi-second rebuild time even on a modest CI machine.
        Assert.True(loaded.Index.RestoredFromStoredTopology, "reopen rebuilt the graph instead of loading stored adjacency pages");
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(3), $"reopen took {clock.Elapsed.TotalSeconds:F2}s");
        Assert.Equal(Records, loaded.Count);
    }

    [Fact]
    public void Replace_RelinksGraphToNewVectorPosition()
    {
        const int Dimensions = 16;
        const int Records = 500;
        const int Queries = 40;
        using var db = FaroDb.InMemory();
        var options = new CollectionOptions<Doc>
        {
            Dimensions = Dimensions,
            Distance = Distance.DotProduct,
            Index = new HnswOptions { M = 8, EfConstruction = 96, EfSearch = 64, Seed = 1234 },
            Serializer = TestFixtures.Serializer(),
            Schema = TestFixtures.Schema(),
            DefaultEfSearch = 64,
        };
        FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", options);

        for (int i = 0; i < Records; i++)
        {
            docs.Upsert(i, DeterministicRandom.UniformVector(Dimensions, DeterministicRandom.DeriveSeed(11, i)), MakeDoc(i));
        }

        var updated = new float[Queries][];
        for (int q = 0; q < Queries; q++)
        {
            // Independent target vector: the old graph links point near the previous position.
            updated[q] = DeterministicRandom.UniformVector(Dimensions, DeterministicRandom.DeriveSeed(99, q));
            docs.Upsert(q, updated[q], MakeDoc(q + 1000));
        }

        int found = 0;
        for (int q = 0; q < Queries; q++)
        {
            SearchHit<int>[] hits = docs.Search(updated[q], 10, efSearch: 64);
            if (hits.Any(h => h.Key == q))
            {
                found++;
            }
        }

        // Hand-derived: after relinking, each moved node is its own nearest neighbour and must be
        // reachable from the entry point; allow a small number of HNSW misses.
        Assert.True(found >= Queries - 4, $"only {found}/{Queries} moved nodes were reachable after Replace");
    }
}
