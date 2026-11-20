using System.Collections.Concurrent;
using Faro;
using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Hnsw;

/// <summary>
/// Fine-grained-locking stress tests: multiple concurrent writers, concurrent readers, and
/// writers racing with readers. Deletes are serialised but still race with searches.
/// </summary>
public class HnswConcurrencyTests
{
    private const int Dim = 24;

    [Fact]
    public async Task ConcurrentWriters_AndReaders_StayConsistent()
    {
        const int writers = 4;
        const int perWriter = 600;

        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = Dim,
            NormalizeOnAdd = false,
            M = 8,
            EfConstruction = 80,
            EfSearch = 48,
            InitialCapacity = writers * perWriter + 64,
            Seed = 0x5eed_d000,
        });

        var errors = new ConcurrentQueue<Exception>();
        var cancellation = TestContext.Current.CancellationToken;
        var query = DeterministicRandom.GaussianVector(Dim, 0x5eed_d001);

        var writerTasks = Enumerable.Range(0, writers).Select(w => Task.Run(() =>
        {
            try
            {
                for (int i = 0; i < perWriter; i++)
                {
                    index.Add(DeterministicRandom.GaussianVector(
                        Dim, DeterministicRandom.DeriveSeed(0x5eed_d002, w * perWriter + i)));
                }
            }
            catch (Exception ex)
            {
                errors.Enqueue(ex);
            }
        }, cancellation)).ToArray();

        var readerTasks = Enumerable.Range(0, 4).Select(_ => Task.Run(() =>
        {
            try
            {
                // Bounded so the readers cannot starve the writers on a small thread pool;
                // they still overlap with the full insert phase.
                for (int iteration = 0; iteration < 3_000; iteration++)
                {
                    var hits = index.Search(query, 10);
                    for (int i = 1; i < hits.Length; i++)
                    {
                        Assert.True(hits[i - 1].Score <= hits[i].Score + 1e-6f, "results must be ordered");
                    }

                    Assert.All(hits, hit =>
                    {
                        Assert.True(hit.Id >= 0);
                        Assert.True(float.IsFinite(hit.Score));
                    });
                }
            }
            catch (Exception ex)
            {
                errors.Enqueue(ex);
            }
        }, cancellation)).ToArray();

        await Task.WhenAll(writerTasks);
        await Task.WhenAll(readerTasks);

        Assert.True(errors.IsEmpty, $"concurrent failures: {string.Join("; ", errors.Select(e => e.ToString()))}");
        Assert.Equal(writers * perWriter, index.Count);
        Assert.Equal(writers * perWriter, index.LiveCount);
    }

    [Fact]
    public async Task ConcurrentSearchesOverFrozenIndex_AreDeterministic()
    {
        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = Dim,
            NormalizeOnAdd = false,
            M = 8,
            EfConstruction = 120,
            EfSearch = 64,
            Seed = 0x5eed_d003,
        });

        for (int id = 0; id < 1_500; id++)
        {
            index.Add(DeterministicRandom.GaussianVector(Dim, DeterministicRandom.DeriveSeed(0x5eed_d004, id)));
        }

        var query = DeterministicRandom.GaussianVector(Dim, 0x5eed_d005);
        var expected = index.Search(query, 10);
        var errors = new ConcurrentQueue<Exception>();
        var options = new ParallelOptions { CancellationToken = TestContext.Current.CancellationToken };

        Parallel.For(0, 8, options, _ =>
        {
            try
            {
                for (int i = 0; i < 150; i++)
                {
                    Assert.Equal(expected, index.Search(query, 10));
                }
            }
            catch (Exception ex)
            {
                errors.Enqueue(ex);
            }
        });

        Assert.True(errors.IsEmpty, $"concurrent failures: {string.Join("; ", errors.Select(e => e.ToString()))}");
    }

    [Fact]
    public async Task WritersReadersAndDeletes_RemainConsistent_AndCompactCleansUp()
    {
        const int total = 1_200;
        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = Dim,
            NormalizeOnAdd = false,
            M = 8,
            EfConstruction = 100,
            EfSearch = 48,
            InitialCapacity = total + 32,
            Seed = 0x5eed_d006,
        });
        for (int id = 0; id < total; id++)
        {
            index.Add(DeterministicRandom.GaussianVector(Dim, DeterministicRandom.DeriveSeed(0x5eed_d007, id)));
        }

        var errors = new ConcurrentQueue<Exception>();
        var cancellation = TestContext.Current.CancellationToken;
        var query = DeterministicRandom.GaussianVector(Dim, 0x5eed_d008);

        var searchers = Enumerable.Range(0, 3).Select(_ => Task.Run(() =>
        {
            try
            {
                for (int iteration = 0; iteration < 2_000; iteration++)
                {
                    foreach (var hit in index.Search(query, 8, efSearch: 64))
                    {
                        Assert.True(hit.Id >= 0 && hit.Id < total + 300);
                        Assert.True(float.IsFinite(hit.Score));
                    }
                }
            }
            catch (Exception ex)
            {
                errors.Enqueue(ex);
            }
        }, cancellation)).ToArray();

        var writer = Task.Run(() =>
        {
            try
            {
                for (int i = 0; i < 300; i++)
                {
                    index.Add(DeterministicRandom.GaussianVector(Dim, DeterministicRandom.DeriveSeed(0x5eed_d009, 100_000 + i)));
                }
            }
            catch (Exception ex)
            {
                errors.Enqueue(ex);
            }
        }, cancellation);

        var deleter = Task.Run(() =>
        {
            try
            {
                for (int i = 0; i < 120; i++)
                {
                    index.Delete(i * 3);
                }
            }
            catch (Exception ex)
            {
                errors.Enqueue(ex);
            }
        }, cancellation);

        await Task.WhenAll(writer, deleter);
        await Task.WhenAll(searchers);

        Assert.True(errors.IsEmpty, $"concurrent failures: {string.Join("; ", errors.Select(e => e.ToString()))}");
        Assert.Equal(total + 300, index.Count);

        index.Compact();
        var compacted = index.CurrentView;
        Assert.Equal(index.LiveCount, Enumerable.Range(0, index.Count).Count(id => compacted.Nodes[id] is { Deleted: false }));

        var view = index.CurrentView;
        for (int id = 0; id < view.Count; id++)
        {
            HnswNode? node = view.Nodes[id];
            if (node is null)
            {
                continue;
            }

            for (int layer = 0; layer <= node.Level; layer++)
            {
                foreach (int link in node.GetLinks(layer))
                {
                    Assert.NotNull(view.Nodes[link]);
                    Assert.False(view.Nodes[link]!.Deleted);
                }
            }
        }
    }
}
