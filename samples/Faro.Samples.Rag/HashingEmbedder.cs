namespace Faro.Samples.Rag;

/// <summary>
/// A deterministic, network-free embedder: tokens are hashed into a fixed-dimensional space
/// with the feature-hashing trick (Weinberger et al., 2009, "Feature Hashing for Large Scale
/// Multitask Learning") using signed hashing, then L2-normalized. No semantics — but stable,
/// offline, and good enough to exercise search end to end.
/// </summary>
public sealed class HashingEmbedder
{
    private const ulong FnvOffsetBasis = 0xCBF2_9CE4_8422_2325UL;
    private const ulong FnvPrime = 0x0000_0100_0000_01B3UL;

    public int Dimensions { get; }

    public HashingEmbedder(int dimensions)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimensions, 1);
        Dimensions = dimensions;
    }

    public float[] Embed(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var vector = new float[Dimensions];
        foreach (var token in Tokenize(text))
        {
            ulong hash = Fnv1a64(token);
            int bucket = (int)(hash % (ulong)Dimensions);

            // A second mix decides the sign, so bucket collisions cancel out on average.
            ulong signMix = (hash * 0x9E37_79B9_7F4A_7C15UL) >> 63;
            vector[bucket] += signMix == 1 ? -1f : 1f;
        }

        Distances.Normalize(vector);
        return vector;
    }

    private static IEnumerable<string> Tokenize(string text)
    {
        var token = new char[64];
        int length = 0;
        foreach (var c in text)
        {
            if (char.IsLetterOrDigit(c))
            {
                if (length == token.Length)
                {
                    Array.Resize(ref token, token.Length * 2);
                }

                token[length++] = char.ToLowerInvariant(c);
            }
            else if (length > 0)
            {
                yield return new string(token, 0, length);
                length = 0;
            }
        }

        if (length > 0)
        {
            yield return new string(token, 0, length);
        }
    }

    private static ulong Fnv1a64(ReadOnlySpan<char> token)
    {
        ulong hash = FnvOffsetBasis;
        foreach (var c in token)
        {
            hash ^= c;
            hash *= FnvPrime;
        }

        return hash;
    }
}
