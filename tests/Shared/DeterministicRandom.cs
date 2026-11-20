namespace Faro.Testing;

/// <summary>
/// Deterministic pseudo-random data for tests and recall suites. Built on splitmix64 so data
/// is identical across processes, target frameworks and .NET versions (unlike
/// <see cref="Random"/>, whose algorithm is not a stability guarantee).
/// </summary>
internal static class DeterministicRandom
{
    /// <summary>Returns the next splitmix64 value and advances <paramref name="state"/>.</summary>
    public static ulong Next(ref ulong state)
    {
        state += 0x9E37_79B9_7F4A_7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D0_49BB_1331_11EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>Returns a uniform double in [0, 1).</summary>
    public static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / (1UL << 53));

    /// <summary>Returns a uniform float in [-1, 1).</summary>
    public static float NextFloat(ref ulong state) => (float)(2.0 * NextDouble(ref state) - 1.0);

    /// <summary>Returns a uniform int in [0, maxExclusive).</summary>
    public static int NextInt(ref ulong state, int maxExclusive) => (int)(Next(ref state) % (ulong)maxExclusive);

    /// <summary>Mixes an index into a base seed to derive per-case seeds.</summary>
    public static ulong DeriveSeed(ulong baseSeed, int index) => baseSeed + (ulong)(uint)index * 0x9E37_79B9_7F4A_7C15UL;

    /// <summary>Returns a vector of uniform floats in [-1, 1).</summary>
    public static float[] UniformVector(int dimensions, ulong seed)
    {
        ulong state = seed;
        var vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++)
        {
            vector[i] = NextFloat(ref state);
        }

        return vector;
    }

    /// <summary>Returns a vector of standard-normal floats (Box-Muller, computed in double).</summary>
    public static float[] GaussianVector(int dimensions, ulong seed)
    {
        ulong state = seed;
        var vector = new float[dimensions];
        for (int i = 0; i < dimensions; i += 2)
        {
            double u1 = 1.0 - NextDouble(ref state);
            double u2 = NextDouble(ref state);
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            vector[i] = (float)(r * Math.Cos(2.0 * Math.PI * u2));
            if (i + 1 < dimensions)
            {
                vector[i + 1] = (float)(r * Math.Sin(2.0 * Math.PI * u2));
            }
        }

        return vector;
    }

    /// <summary>Returns a standard-normal vector scaled to unit L2 length.</summary>
    public static float[] UnitVector(int dimensions, ulong seed)
    {
        var vector = GaussianVector(dimensions, seed);
        double sumSquares = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares += (double)vector[i] * vector[i];
        }

        if (sumSquares == 0)
        {
            // Absurdly unlikely, but keep determinism: use a fallback basis vector.
            vector[0] = 1f;
            return vector;
        }

        float inverse = (float)(1.0 / Math.Sqrt(sumSquares));
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= inverse;
        }

        return vector;
    }

    /// <summary>Returns packed words with uniformly random bits.</summary>
    public static ulong[] PackedBits(int words, ulong seed)
    {
        ulong state = seed;
        var packed = new ulong[words];
        for (int i = 0; i < words; i++)
        {
            packed[i] = Next(ref state);
        }

        return packed;
    }
}
