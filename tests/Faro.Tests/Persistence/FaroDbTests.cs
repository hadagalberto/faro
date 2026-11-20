using Faro.Filtering;
using Faro.Persistence;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Persistence;

public class FaroDbTests : PersistenceTestBase
{
    [Fact]
    public void Checkpoint_RoundTripsVectorsPayloadsFiltersAndHybrid()
    {
        string path = DbPath();
        using (var db = FaroDb.Open(path, new FaroOptions { Durability = Durability.Batched }))
        {
            FaroDbCollection<Doc> written = db.GetOrCreateCollection("docs", Options());
            for (int i = 0; i < 20; i++)
            {
                written.Upsert(i, OneHot(i % 16), MakeDoc(i));
            }

            db.Checkpoint();
        }

        Assert.True(File.Exists(path));
        Assert.Equal(16, new FileInfo(path + WalFile.Suffix).Length); // WAL header only
        Assert.False(File.Exists(path + ".tmp"));

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> loaded = reopened.GetOrCreateCollection("docs", Options());
        Assert.Equal(20, loaded.Count);
        Assert.True(loaded.Index.UsesMappedVectors);
        var stored = new float[16];
        Assert.True(loaded.Index.TryGetVector(3, stored));
        Assert.Equal(1f, stored[3]);
        Assert.Equal(0f, stored[0]);
        Assert.True(loaded.TryGet(7, out Doc? payload));
        Assert.Equal(MakeDoc(7).Text, payload!.Text);
        Assert.Equal(2027, payload.Year);
        Assert.Contains(7, loaded.Keys);

        SearchResult[] raw = loaded.Index.Search(OneHot(3), 3, 32);
        Assert.Equal(3, raw[0].Id);
        SearchHit<int>[] hits = loaded.Search(OneHot(3), 3);
        Assert.Equal(3, hits[0].Key);
        Assert.Equal(1f, hits[0].Score, 5);
        Assert.All(hits, h => Assert.True(loaded.TryGet(h.Key, out _)));

        SearchHit<int>[] filtered = loaded.Search(OneHot(4), 10, Filter.Eq("tenant", "acme"));
        Assert.NotEmpty(filtered);
        Assert.All(filtered, h =>
        {
            Assert.True(loaded.TryGet(h.Key, out Doc? doc));
            Assert.Equal("acme", doc!.Tenant);
        });

        HybridHit<int>[] hybrid = loaded.HybridSearch(OneHot(4), "gato", 10);
        Assert.NotEmpty(hybrid);
        Assert.Contains(hybrid, h => h.TextRank > 0);
    }

    [Fact]
    public void WalOnly_ReplaysUpsertsDeletesAndUpdates()
    {
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> written = db.GetOrCreateCollection("docs", Options());
            for (int i = 0; i < 6; i++)
            {
                written.Upsert(i, OneHot(i), MakeDoc(i));
            }

            Assert.True(written.Delete(2));
            Assert.False(written.Delete(2));
            written.Upsert(2, OneHot(2), new Doc { Tenant = "new", Text = "updated" });
        }

        Assert.False(File.Exists(path)); // no checkpoint was created

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> loaded = reopened.GetOrCreateCollection("docs", Options());
        Assert.Equal(6, loaded.Count);
        Assert.True(loaded.TryGet(2, out Doc? updated));
        Assert.Equal("new", updated!.Tenant);
        Assert.Equal("updated", updated.Text);
        Assert.True(reopened.GetCollection<Doc>("docs")!.TryGet(3, out Doc? d3));
        Assert.Equal(2023, d3!.Year);
        Assert.Equal(2, loaded.Search(OneHot(2), 1)[0].Key);
    }

    [Theory]
    [InlineData(Durability.Always)]
    [InlineData(Durability.Batched)]
    [InlineData(Durability.None)]
    public void DurabilityModes_ReplayCommittedRecords(Durability durability)
    {
        string path = DbPath($"durability-{durability}.faro");
        using (var db = FaroDb.Open(path, new FaroOptions { Durability = durability }))
        {
            FaroDbCollection<Doc> written = db.GetOrCreateCollection("docs", Options());
            written.Upsert(1, OneHot(1), MakeDoc(1));
            written.Upsert(2, OneHot(2), MakeDoc(2));
            db.Flush();
        }

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> loaded = reopened.GetOrCreateCollection("docs", Options());
        Assert.Equal(2, loaded.Count);
        Assert.True(loaded.ContainsKey(1) && loaded.ContainsKey(2));
    }

    [Fact]
    public void InMemoryMode_UsesNoFilesAndRejectsCheckpoint()
    {
        using var db = FaroDb.InMemory();
        Assert.True(db.IsInMemory);
        Assert.Null(db.Path);
        FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options());
        docs.Upsert(1, OneHot(1), MakeDoc(1));
        Assert.Equal(1, docs.Count);
        Assert.Throws<InvalidOperationException>(() => db.Checkpoint());
        Assert.Empty(Directory.GetFileSystemEntries(Root));
    }

    [Fact]
    public async Task AsyncApi_RoundTripsAndHonoursCancellation()
    {
        CancellationToken token = TestContext.Current.CancellationToken;
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> written = db.GetOrCreateCollection("docs", Options());
            await written.UpsertAsync(1, OneHot(1), MakeDoc(1), token);
            await written.UpsertBatchAsync(
                Enumerable.Range(2, 4).Select(i => (i, OneHot(i), MakeDoc(i))), token);
            Assert.Equal(5, written.Count);

            SearchHit<int>[] hits = await written.SearchAsync(OneHot(3), 2, cancellationToken: token);
            Assert.Equal(3, hits[0].Key);
            HybridHit<int>[] hybrid = await written.HybridSearchAsync(OneHot(3), "gato", 5, cancellationToken: token);
            Assert.NotEmpty(hybrid);
            Assert.True(await written.DeleteAsync(4, token));
            await db.FlushAsync(token);
            await db.CheckpointAsync(token);
        }

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> loaded = reopened.GetOrCreateCollection("docs", Options());
        Assert.Equal(4, loaded.Count);
        Assert.False(loaded.ContainsKey(4));

        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => loaded.UpsertAsync(9, OneHot(9), MakeDoc(9), cancelled.Token));
    }

    [Fact]
    public void UpsertBatch_ParallelInsertsAllRecords()
    {
        using var db = FaroDb.InMemory();
        FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options());
        docs.UpsertBatch(Enumerable.Range(0, 500)
            .Select(i => (i, OneHot(i % 16), MakeDoc(i))));

        Assert.Equal(500, docs.Count);
        for (int i = 0; i < 500; i++)
        {
            Assert.True(docs.TryGet(i, out Doc? doc));
            Assert.Equal(MakeDoc(i).Year, doc!.Year);
        }
    }

    [Fact]
    public void GenericKeys_RoundTripThroughCheckpointAndWal()
    {
        string path = DbPath();
        Guid guid = Guid.Parse("6f9619ff-8b86-d011-b42d-00c04fc964ff");
        using (var db = FaroDb.Open(path))
        {
            var writtenLongs = db.GetOrCreateCollection<long, Doc>("longs", Options());
            var writtenGuids = db.GetOrCreateCollection<Guid, Doc>("guids", Options());
            var writtenStrings = db.GetOrCreateCollection<string, Doc>("strings", Options());
            writtenLongs.Upsert(9_000_000_000L, OneHot(1), MakeDoc(1));
            writtenGuids.Upsert(guid, OneHot(2), MakeDoc(2));
            writtenStrings.Upsert("tenant/key", OneHot(3), MakeDoc(3));
            db.Checkpoint();
            writtenLongs.Upsert(9_000_000_001L, OneHot(4), MakeDoc(4)); // post-checkpoint WAL record
        }

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<long, Doc> longs = reopened.GetOrCreateCollection<long, Doc>("longs", Options());
        FaroDbCollection<Guid, Doc> guids = reopened.GetOrCreateCollection<Guid, Doc>("guids", Options());
        FaroDbCollection<string, Doc> strings = reopened.GetOrCreateCollection<string, Doc>("strings", Options());

        Assert.Equal(2, longs.Count);
        Assert.True(longs.TryGet(9_000_000_000L, out Doc? longDoc));
        Assert.Equal(2021, longDoc!.Year);
        Assert.True(guids.TryGet(guid, out Doc? guidDoc));
        Assert.Equal(2022, guidDoc!.Year);
        Assert.Equal("guids", guids.Name);
        Assert.True(strings.TryGet("tenant/key", out Doc? stringDoc));
        Assert.Equal(2023, stringDoc!.Year);
        Assert.Equal("tenant/key", strings.Search(OneHot(3), 1)[0].Key);
    }

    [Fact]
    public void InvalidOptionsAndMismatches_AreRejected()
    {
        using (var db = FaroDb.InMemory())
        {
            Assert.Throws<NotSupportedException>(() =>
                db.GetOrCreateCollection<double, Doc>("bad-key", Options()));
            Assert.Throws<ArgumentException>(() =>
                db.GetOrCreateCollection("no-serializer", new CollectionOptions<Doc> { Dimensions = 4 }));

            FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options());
            docs.Upsert(1, OneHot(1), MakeDoc(1));
            Assert.Throws<InvalidOperationException>(() =>
                db.GetOrCreateCollection("docs", new CollectionOptions<Doc>
                {
                    Dimensions = 8,
                    Schema = TestFixtures.Schema(),
                    Serializer = TestFixtures.Serializer(),
                }));
        }

        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            db.GetOrCreateCollection("docs", Options());
        }

        string corruptPath = DbPath("corrupt.faro");
        File.WriteAllText(corruptPath, "this is not a Faro checkpoint");
        Assert.Throws<InvalidDataException>(() =>
        {
            using var corrupt = FaroDb.Open(corruptPath);
        });
    }

    [Fact]
    public void RepeatedCheckpoints_ApplyUpdatesAndDeletesAfterReopen()
    {
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> written = db.GetOrCreateCollection("docs", Options());
            written.Upsert(1, OneHot(1), MakeDoc(1));
            written.Upsert(2, OneHot(2), MakeDoc(2));
            db.Checkpoint();

            written.Upsert(1, OneHot(5), MakeDoc(50));
            Assert.True(written.Delete(2));
            db.Checkpoint();
        }

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> loaded = reopened.GetOrCreateCollection("docs", Options());
        Assert.Equal(1, loaded.Count);
        Assert.Equal(1, loaded.Search(OneHot(5), 1)[0].Key);
        Assert.True(loaded.TryGet(1, out Doc? updated));
        Assert.Equal(2070, updated!.Year);
        Assert.Contains("gato", updated.Text);
    }

    [Fact]
    public void HammingAndQuantizedCollections_RoundTripThroughCheckpoint()
    {
        string path = DbPath();
        var hammingOptions = new CollectionOptions<Doc>
        {
            Dimensions = 64,
            Distance = Distance.Hamming,
            Index = new HnswOptions { M = 8, EfConstruction = 64, EfSearch = 32, Seed = 7 },
            Serializer = TestFixtures.Serializer(),
            Schema = TestFixtures.Schema(),
        };
        var quantizedOptions = new CollectionOptions<Doc>
        {
            Dimensions = 16,
            Distance = Distance.DotProduct,
            Index = new HnswOptions { M = 8, EfConstruction = 64, EfSearch = 32, Seed = 11 },
            Quantization = global::Faro.Quantization.Int8(rescore: 4),
            Serializer = TestFixtures.Serializer(),
            Schema = TestFixtures.Schema(),
        };

        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> binary = db.GetOrCreateCollection("binary", hammingOptions);
            FaroDbCollection<Doc> quantized = db.GetOrCreateCollection("quantized", quantizedOptions);
            for (int i = 0; i < 12; i++)
            {
                binary.Upsert(i, Signs(i), MakeDoc(i));
                quantized.Upsert(i, OneHot(i), MakeDoc(i));
            }

            db.Checkpoint();
        }

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> loadedBinary = reopened.GetOrCreateCollection("binary", hammingOptions);
        FaroDbCollection<Doc> loadedQuantized = reopened.GetOrCreateCollection("quantized", quantizedOptions);
        Assert.Equal(12, loadedBinary.Count);
        Assert.Equal(12, loadedQuantized.Count);
        Assert.True(loadedBinary.Index.UsesMappedVectors);
        var unpacked = new float[64];
        Assert.True(loadedBinary.Index.TryGetVector(3, unpacked));
        Assert.Equal(Signs(3), unpacked);
        Assert.Equal(3, loadedBinary.Index.Search(Signs(3), 1, 32)[0].Id);
        Assert.Equal(3, loadedBinary.Search(Signs(3), 1)[0].Key);
        Assert.Equal(4, loadedQuantized.Search(OneHot(4), 1)[0].Key);
    }

    private static float[] Signs(int seed, int dimensions = 64)
    {
        var vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++)
        {
            uint hash = unchecked((uint)seed * 2654435761u) ^ unchecked((uint)i * 2246822519u);
            hash ^= hash >> 13;
            hash *= 1274126177u;
            hash ^= hash >> 16;
            vector[i] = (hash & 1) == 0 ? 1f : -1f;
        }

        return vector;
    }

    [Fact]
    public void CheckpointBytes_AreDeterministic()
    {
        string first = DbPath("first.faro");
        string second = DbPath("second.faro");
        foreach (string path in new[] { first, second })
        {
            using var db = FaroDb.Open(path);
            FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options());
            for (int i = 0; i < 12; i++)
            {
                docs.Upsert(i, OneHot(i), MakeDoc(i));
            }

            db.Checkpoint();
        }

        Assert.Equal(File.ReadAllBytes(first), File.ReadAllBytes(second));
    }
}
