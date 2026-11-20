namespace Faro.Internal;

/// <summary>
/// Deterministic, dependency-free uniform random values keyed by a seed and an integer index.
/// Built on splitmix64 so the stream is bit-identical across processes, target frameworks and
/// .NET versions — HNSW uses it to assign node levels when a fixed <c>Seed</c> is requested.
/// </summary>
internal static class SeededUniform
{
    private const ulong Gamma = 0x9E37_79B9_7F4A_7C15UL;

    /// <summary>Returns the splitmix64 mix of a seed combined with an index.</summary>
    public static ulong Mix(ulong seed, int index)
    {
        ulong z = seed + (ulong)(uint)index * Gamma;
        z = (z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D0_49BB_1331_11EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>Returns a uniform value in the open-closed interval (0, 1].</summary>
    public static double Unit(ulong seed, int index) =>
        1.0 - (Mix(seed, index) >> 11) * (1.0 / (1UL << 53));
}
