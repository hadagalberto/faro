using System.Collections.Concurrent;
using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Flat;

public class FlatConcurrencyTests
{
    [Fact]
    public async Task SearchesWhileSingleWriterAdds_NeverThrowAndStayConsistent()
    {
        const int initial = 500;
        const int added = 20_000;

        var index = new FlatIndex(new FlatOptions
        {
            Dimensions = 32,
            Distance = Distance.Cosine,
            InitialCapacity = 32_768, // avoid growth swaps competing with the search loop
            ParallelThreshold = 512,  // searchers flip to the parallel path mid-test
        });

        for (int id = 0; id < initial; id++)
        {
            index.Add(DeterministicRandom.GaussianVector(32, DeterministicRandom.DeriveSeed(0x5eed_b000, id)));
        }

        var query = DeterministicRandom.GaussianVector(32, 0x5eed_b001);
        var normalizedQuery = (float[])query.Clone();
        Distances.Normalize(normalizedQuery);
        var errors = new ConcurrentQueue<Exception>();
        var writerDone = new ManualResetEventSlim(false);
        var stored = new float[32];
        var cancellation = TestContext.Current.CancellationToken;

        var writer = Task.Run(() =>
        {
            try
            {
                for (int i = 0; i < added; i++)
                {
                    index.Add(DeterministicRandom.GaussianVector(32, DeterministicRandom.DeriveSeed(0x5eed_b002, initial + i)));
                }
            }
            catch (Exception ex)
            {
                errors.Enqueue(ex);
            }
            finally
            {
                writerDone.Set();
            }
        }, cancellation);

        var searchers = Enumerable.Range(0, 4).Select(_ => Task.Run(() =>
        {
            try
            {
                long searches = 0;
                while (!writerDone.IsSet || searches < 10)
                {
                    var hits = index.Search(query, 8);
                    Assert.True(hits.Length <= 8);

                    foreach (var hit in hits)
                    {
                        Assert.True(hit.Id >= 0, "ids must be non-negative");
                        Assert.True(float.IsFinite(hit.Score), "scores must be finite");

                        // The writer never mutates published vectors, so recomputing the score
                        // from the stored form and the normalized query must reproduce it exactly.
                        Assert.True(index.TryGetVector(hit.Id, stored));
                        float recomputed = 1f - Distances.Dot(normalizedQuery, stored);
                        FloatAssertions.Equal(recomputed, hit.Score, 0f);
                    }

                    searches++;
                }
            }
            catch (Exception ex)
            {
                errors.Enqueue(ex);
            }
        }, cancellation)).ToArray();

        await Task.WhenAll(searchers.Append(writer).ToArray());

        Assert.True(errors.IsEmpty, $"concurrent failures: {string.Join("; ", errors.Select(e => e.Message))}");
        Assert.Equal(initial + added, index.Count);
    }

    [Fact]
    public void ConcurrentSearchesOverAFrozenIndex_AreConsistent()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = 64, Distance = Distance.Euclidean, NormalizeOnAdd = false });
        for (int id = 0; id < 4_000; id++)
        {
            index.Add(DeterministicRandom.GaussianVector(64, DeterministicRandom.DeriveSeed(0x5eed_b003, id)));
        }

        var query = DeterministicRandom.GaussianVector(64, 0x5eed_b004);
        var expected = index.Search(query, 10);

        var errors = new ConcurrentQueue<Exception>();
        var options = new ParallelOptions { CancellationToken = TestContext.Current.CancellationToken };
        Parallel.For(0, 8, options, _ =>
        {
            try
            {
                for (int i = 0; i < 200; i++)
                {
                    var results = index.Search(query, 10);
                    Assert.Equal(expected, results);
                }
            }
            catch (Exception ex)
            {
                errors.Enqueue(ex);
            }
        });

        Assert.True(errors.IsEmpty, $"concurrent failures: {string.Join("; ", errors.Select(e => e.Message))}");
    }
}
