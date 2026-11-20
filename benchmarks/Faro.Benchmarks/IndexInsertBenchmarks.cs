using BenchmarkDotNet.Attributes;
using Faro;
using Faro.Testing;

namespace Faro.Benchmarks;

/// <summary>
/// Single-vector insert throughput for every index/quantization combination. Each invocation
/// builds a fresh index so the cost of growing it is included; the <c>OperationsPerInvoke</c>
/// benchmark property makes BenchmarkDotNet report time and allocated bytes per inserted vector.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public class IndexInsertBenchmarks
{
    private const int Count = 5_000;
    private const int Dimensions = 128;

    private float[][] _data = null!;

    [GlobalSetup]
    public void Setup()
    {
        _data = new float[Count][];
        for (int i = 0; i < Count; i++)
        {
            _data[i] = DeterministicRandom.UnitVector(Dimensions, DeterministicRandom.DeriveSeed(0x5eed_e001, i));
        }
    }

    [Benchmark(Baseline = true, OperationsPerInvoke = Count)]
    public int FlatInsert()
    {
        var index = new FlatIndex(new FlatOptions { Dimensions = Dimensions, InitialCapacity = Count });
        return AddAll(index);
    }

    [Benchmark(OperationsPerInvoke = Count)]
    public int FlatInsertInt8()
    {
        var index = new FlatIndex(new FlatOptions
        {
            Dimensions = Dimensions,
            InitialCapacity = Count,
            Quantization = Faro.Quantization.Int8(rescore: 4),
        });
        return AddAll(index);
    }

    [Benchmark(OperationsPerInvoke = Count)]
    public int FlatInsertBinary()
    {
        var index = new FlatIndex(new FlatOptions
        {
            Dimensions = Dimensions,
            InitialCapacity = Count,
            Quantization = Faro.Quantization.Binary(rescore: 8),
        });
        return AddAll(index);
    }

    [Benchmark(OperationsPerInvoke = Count)]
    public int HnswInsert()
    {
        var index = NewHnsw(null);
        return AddAll(index);
    }

    [Benchmark(OperationsPerInvoke = Count)]
    public int HnswInsertInt8()
    {
        var index = NewHnsw(Faro.Quantization.Int8(rescore: 4));
        return AddAll(index);
    }

    [Benchmark(OperationsPerInvoke = Count)]
    public int HnswInsertBinary()
    {
        var index = NewHnsw(Faro.Quantization.Binary(rescore: 8));
        return AddAll(index);
    }

    private static HnswIndex NewHnsw(QuantizationOptions? quantization) => new(new HnswOptions
    {
        Dimensions = Dimensions,
        InitialCapacity = Count,
        Distance = Distance.Cosine,
        M = 16,
        EfConstruction = 128,
        EfSearch = 64,
        Seed = 0x5eed_e002,
        Quantization = quantization,
    });

    private int AddAll(FlatIndex index)
    {
        for (int i = 0; i < Count; i++)
        {
            index.Add(_data[i]);
        }

        return index.Count;
    }

    private int AddAll(HnswIndex index)
    {
        for (int i = 0; i < Count; i++)
        {
            index.Add(_data[i]);
        }

        return index.Count;
    }
}
