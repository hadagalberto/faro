using BenchmarkDotNet.Attributes;
using Faro;
using Faro.Testing;

namespace Faro.Benchmarks;

/// <summary>
/// Exact Flat search: SIMD sequential vs parallel vs a naive scalar brute force with a sort.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public class FlatSearchBenchmarks
{
    [Params(20_000)]
    public int Count { get; set; }

    [Params(768)]
    public int Dimensions { get; set; }

    private FlatIndex _sequential = null!;
    private FlatIndex _parallel = null!;
    private float[][] _data = null!;
    private float[] _query = null!;

    [GlobalSetup]
    public void Setup()
    {
        _data = new float[Count][];
        _sequential = new FlatIndex(new FlatOptions { Dimensions = Dimensions, Distance = Distance.Cosine, ParallelThreshold = int.MaxValue });
        _parallel = new FlatIndex(new FlatOptions { Dimensions = Dimensions, Distance = Distance.Cosine, ParallelThreshold = 1 });
        for (int id = 0; id < Count; id++)
        {
            _data[id] = DeterministicRandom.UnitVector(Dimensions, DeterministicRandom.DeriveSeed(0x5eed_d010, id));
            _sequential.Add(_data[id]);
            _parallel.Add(_data[id]);
        }

        _query = DeterministicRandom.UnitVector(Dimensions, 0x5eed_d011);
    }

    [Benchmark]
    public SearchResult[] FlatSequential() => _sequential.Search(_query, 10);

    [Benchmark]
    public SearchResult[] FlatParallel() => _parallel.Search(_query, 10);

    [Benchmark]
    public SearchResult[] NaiveScalarBruteForce()
    {
        var scored = new (int Id, float Score)[Count];
        for (int id = 0; id < Count; id++)
        {
            scored[id] = (id, Distances.CosineScalar(_query, _data[id]));
        }

        Array.Sort(scored, static (x, y) => x.Score.CompareTo(y.Score));
        var results = new SearchResult[10];
        for (int i = 0; i < 10; i++)
        {
            results[i] = new SearchResult(scored[i].Id, scored[i].Score);
        }

        return results;
    }
}
