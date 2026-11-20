using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Hnsw;

public class HnswZeroAllocationTests
{
    private const int Count = 2_048;
    private const int Dim = 64;

    [Theory]
    [InlineData(Distance.Cosine)]
    [InlineData(Distance.DotProduct)]
    [InlineData(Distance.Euclidean)]
    [InlineData(Distance.Hamming)]
    public void Search_IsAllocationFreeInSteadyState(Distance distance)
    {
        var index = new HnswIndex(new HnswOptions
        {
            Dimensions = Dim,
            Distance = distance,
            NormalizeOnAdd = distance == Distance.Cosine ? true : false,
            M = 12,
            EfConstruction = 80,
            EfSearch = 48,
            Seed = 0x5eed_9000,
        });

        for (int id = 0; id < Count; id++)
        {
            index.Add(DeterministicRandom.UniformVector(Dim, DeterministicRandom.DeriveSeed(0x5eed_9001, id)));
        }

        var query = DeterministicRandom.UniformVector(Dim, 0x5eed_9002);
        var buffer = new SearchResult[16];

        // Warm the JIT and every ArrayPool bucket the search touches.
        for (int i = 0; i < 3_000; i++)
        {
            index.Search(query, 16, buffer);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 3_000; i++)
        {
            index.Search(query, 16, buffer);
        }

        long after = GC.GetAllocatedBytesForCurrentThread();
        Assert.Equal(before, after);
    }
}
