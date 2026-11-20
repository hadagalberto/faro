using Faro.Testing;
using Faro.Tests.Persistence;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Collection;

/// <summary>
/// Concurrent upserts of the same key used to race TryGetValue/AddCore, leaving orphaned live
/// nodes (200 keys produced ~1593 nodes and only 8 distinct keys in the top 50). Expected values
/// here are arithmetic: one live node per distinct key.
/// </summary>
public class ConcurrentUpsertTests : PersistenceTestBase
{
    [Fact]
    public void ConcurrentUpsert_SameKey_IsAtomic()
    {
        const int Keys = 200;
        const int Workers = 8;
        const int Dimensions = 32;
        using var db = FaroDb.InMemory();
        FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options(Dimensions));
        float[][] vectors = Enumerable.Range(0, Keys)
            .Select(key => DeterministicRandom.UniformVector(Dimensions, DeterministicRandom.DeriveSeed(0xBEEF, key)))
            .ToArray();

        Parallel.For(0, Workers, worker =>
        {
            var items = new List<(int Key, float[] Vector, Doc Payload)>(Keys);
            for (int key = 0; key < Keys; key++)
            {
                // Same key in every worker, same vector -> last writer wins, no new node.
                items.Add((key, vectors[key], MakeDoc(key + worker * Keys)));
            }

            docs.UpsertBatch(items);
        });

        // Hand-derived: exactly one node and one key mapping per key, whatever the interleaving.
        Assert.Equal(Keys, docs.Count);
        Assert.Equal(Keys, docs.Keys.Count);
        Assert.Equal(Keys, docs.Index.LiveCount);
        for (int key = 0; key < Keys; key++)
        {
            Assert.True(docs.TryGet(key, out Doc? payload));
            // Hand-derived: the winner is one of the workers that wrote this key.
            Assert.Contains(payload!.Year, Enumerable.Range(0, Workers).Select(w => 2020 + key + w * Keys));
        }

        SearchHit<int>[] top = docs.Search(vectors[0], 50);
        Assert.Equal(50, top.Length);
        Assert.Equal(50, top.Select(hit => hit.Key).Distinct().Count());
    }
}
