using BenchmarkDotNet.Attributes;
using Faro;
using Faro.Testing;

namespace Faro.Benchmarks;

/// <summary>
/// Raw distance kernel throughput: the hot inner loop of every search.
/// </summary>
[MemoryDiagnoser]
public class DistanceKernelsBenchmarks
{
    [Params(128, 768)]
    public int Dimensions { get; set; }

    private float[] _a = null!;
    private float[] _b = null!;
    private ulong[] _packedA = null!;
    private ulong[] _packedB = null!;

    [GlobalSetup]
    public void Setup()
    {
        _a = DeterministicRandom.GaussianVector(Dimensions, 0x5eed_d001);
        _b = DeterministicRandom.GaussianVector(Dimensions, 0x5eed_d002);
        _packedA = BitVectors.PackSignBits(_a);
        _packedB = BitVectors.PackSignBits(_b);
    }

    [Benchmark(Baseline = true)]
    public float DotScalar() => Distances.DotScalar(_a, _b);

    [Benchmark]
    public float Dot() => Distances.Dot(_a, _b);

    [Benchmark]
    public float Cosine() => Distances.Cosine(_a, _b);

    [Benchmark]
    public float EuclideanSquared() => Distances.EuclideanSquared(_a, _b);

    [Benchmark]
    public int Hamming() => Distances.Hamming(_packedA, _packedB);
}
