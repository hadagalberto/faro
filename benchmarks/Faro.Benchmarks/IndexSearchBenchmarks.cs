using BenchmarkDotNet.Attributes;
using Faro;
using Faro.Testing;

namespace Faro.Benchmarks;

/// <summary>
/// Search latency (mean column) and throughput (ops/s column) for Flat and HNSW, each without
/// quantization, with int8 + rescoring and with binary + rescoring. The allocating overload is
/// measured because it is the one applications call; the buffered HNSW benchmark isolates the
/// allocation-free routing cost. <see cref="MemoryDiagnoser"/> reports the per-search allocation.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public class IndexSearchBenchmarks
{
    private const int Count = 20_000;
    private const int Dimensions = 128;
    private const int K = 10;

    private FlatIndex _flat = null!;
    private FlatIndex _flatInt8 = null!;
    private FlatIndex _flatBinary = null!;
    private HnswIndex _hnsw = null!;
    private HnswIndex _hnswInt8 = null!;
    private HnswIndex _hnswBinary = null!;
    private float[] _query = null!;
    private SearchResult[] _buffer = null!;

    [GlobalSetup]
    public void Setup()
    {
        var data = new float[Count][];
        for (int i = 0; i < Count; i++)
        {
            data[i] = DeterministicRandom.UnitVector(Dimensions, DeterministicRandom.DeriveSeed(0x5eed_e101, i));
        }

        _query = DeterministicRandom.UnitVector(Dimensions, 0x5eed_e102);
        _buffer = new SearchResult[K];

        _flat = new FlatIndex(Flat());
        _flatInt8 = new FlatIndex(Flat(Faro.Quantization.Int8(rescore: 4)));
        _flatBinary = new FlatIndex(Flat(Faro.Quantization.Binary(rescore: 8)));
        _hnsw = new HnswIndex(Hnsw(null));
        _hnswInt8 = new HnswIndex(Hnsw(Faro.Quantization.Int8(rescore: 4)));
        _hnswBinary = new HnswIndex(Hnsw(Faro.Quantization.Binary(rescore: 8)));

        foreach (float[] vector in data)
        {
            _flat.Add(vector);
            _flatInt8.Add(vector);
            _flatBinary.Add(vector);
            _hnsw.Add(vector);
            _hnswInt8.Add(vector);
            _hnswBinary.Add(vector);
        }

        // Trigger lazy codec training outside the measured region.
        _flatInt8.Search(_query, K);
        _flatBinary.Search(_query, K);
        _hnswInt8.Search(_query, K);
        _hnswBinary.Search(_query, K);
    }

    [Benchmark(Baseline = true)]
    public SearchResult[] FlatSearch() => _flat.Search(_query, K);

    [Benchmark]
    public SearchResult[] FlatInt8Search() => _flatInt8.Search(_query, K);

    [Benchmark]
    public SearchResult[] FlatBinarySearch() => _flatBinary.Search(_query, K);

    [Benchmark]
    public SearchResult[] HnswSearch() => _hnsw.Search(_query, K);

    [Benchmark]
    public SearchResult[] HnswInt8Search() => _hnswInt8.Search(_query, K);

    [Benchmark]
    public SearchResult[] HnswBinarySearch() => _hnswBinary.Search(_query, K);

    [Benchmark]
    public int HnswSearchBuffered() => _hnsw.Search(_query, K, _buffer);

    private static FlatOptions Flat(QuantizationOptions? quantization = null) => new()
    {
        Dimensions = Dimensions,
        Distance = Distance.Cosine,
        InitialCapacity = Count,
        ParallelThreshold = int.MaxValue,
        Quantization = quantization,
    };

    private static HnswOptions Hnsw(QuantizationOptions? quantization) => new()
    {
        Dimensions = Dimensions,
        Distance = Distance.Cosine,
        InitialCapacity = Count,
        M = 16,
        EfConstruction = 128,
        EfSearch = 64,
        Seed = 0x5eed_e103,
        Quantization = quantization,
    };
}
