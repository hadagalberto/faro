using Faro.Filtering;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Persistence;

/// <summary>
/// Searches used to read memory-mapped vectors without the database gate. A concurrent checkpoint
/// disposed the mapping, producing ObjectDisposedException and potentially reading unmapped memory.
/// </summary>
public class SearchCheckpointConcurrencyTests : PersistenceTestBase
{
    [Fact]
    public void ConcurrentSearchAndCheckpoint_NeverThrows()
    {
        const int Dimensions = 32;
        const int Records = 400;
        string path = DbPath();

        using var db = FaroDb.Open(path);
        FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options(Dimensions));
        for (int i = 0; i < Records; i++)
        {
            docs.Upsert(i, OneHot(i % Dimensions, Dimensions), MakeDoc(i));
        }

        using var stop = new CancellationTokenSource();
        var errors = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
        var readers = new List<Task>();
        for (int worker = 0; worker < 4; worker++)
        {
            readers.Add(Task.Run(() =>
            {
                var query = OneHot(0, Dimensions);
                while (!stop.IsCancellationRequested)
                {
                    try
                    {
                        SearchHit<int>[] hits = docs.Search(query, 10);
                        Assert.NotEmpty(hits);
                        _ = docs.HybridSearch(query, "gato", 10);
                        Assert.NotEmpty(docs.GetByFilter(Filter.Eq("tenant", "acme"), 10));
                    }
                    catch (Exception ex)
                    {
                        errors.Enqueue(ex);
                        return;
                    }
                }
            }, TestContext.Current.CancellationToken));
        }

        for (int i = 0; i < 4; i++)
        {
            db.Checkpoint();
        }

        stop.Cancel();
        Assert.All(readers, task => task.Wait(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken));

        // Hand-derived: a checkpoint never invalidates a search in flight, so no reader failed.
        Assert.True(errors.IsEmpty, errors.TryDequeue(out Exception? first) ? first.ToString() : "no errors");
        Assert.Equal(Records, docs.Count);
    }
}
