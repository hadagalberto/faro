using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Flat;

[Collection("ZeroAllocation")]
public class FlatZeroAllocationTests
{
    private const int Count = 2048;
    private const int Dim = 64;

    [Theory]
    [InlineData(Distance.Cosine)]
    [InlineData(Distance.DotProduct)]
    [InlineData(Distance.Euclidean)]
    [InlineData(Distance.Hamming)]
    public void SequentialSearch_IsAllocationFreeInSteadyState(Distance distance)
    {
        // Force the sequential path (threshold far above Count) so the measurement is not
        // polluted by Parallel.For machinery; the parallel path is a throughput path and is
        // validated for equivalence elsewhere.
        var index = new FlatIndex(new FlatOptions
        {
            Dimensions = Dim,
            Distance = distance,
            NormalizeOnAdd = distance == Distance.DotProduct ? false : null,
            ParallelThreshold = int.MaxValue,
        });

        for (int id = 0; id < Count; id++)
        {
            index.Add(DeterministicRandom.UniformVector(Dim, DeterministicRandom.DeriveSeed(0x5eed_a000, id)));
        }

        var query = DeterministicRandom.UniformVector(Dim, 0x5eed_a001);
        var buffer = new SearchResult[16];

        ZeroAllocationAssert.Run(() => index.Search(query, 16, buffer));
    }
}
