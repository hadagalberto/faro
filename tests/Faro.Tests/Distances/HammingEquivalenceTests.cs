using System.Numerics;
using Faro;
using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.DistanceTests;

public class HammingEquivalenceTests
{
    public static TheoryData<int> WordCounts { get; } = new()
    {
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    };

    [Theory]
    [MemberData(nameof(WordCounts))]
    public void Hamming_AllVectorPaths_MatchScalarReference(int words)
    {
        var a = DeterministicRandom.PackedBits(words, DeterministicRandom.DeriveSeed(0x5eed_3001, words));
        var b = DeterministicRandom.PackedBits(words, DeterministicRandom.DeriveSeed(0x5eed_3002, words));

        int reference = Distances.HammingScalar(a, b);

        Assert.Equal(reference, Distances.Hamming(a, b));
        Assert.Equal(reference, BitKernels.HammingScalar(a, b));
        Assert.Equal(reference, BitKernels.HammingV128(a, b));
        Assert.Equal(reference, BitKernels.HammingV256(a, b));
        Assert.Equal(reference, BitKernels.HammingV512(a, b));
    }

    [Theory]
    [MemberData(nameof(WordCounts))]
    public void Hamming_CountsEveryBitExactlyOnce(int words)
    {
        // Complement every bit: distance must be 64 * words, counting each bit exactly once.
        var a = DeterministicRandom.PackedBits(words, DeterministicRandom.DeriveSeed(0x5eed_3003, words));
        var b = a.Select(w => ~w).ToArray();

        Assert.Equal(64 * words, Distances.Hamming(a, b));
        Assert.Equal(64 * words, Distances.HammingScalar(a, b));
    }

    [Fact]
    public void Hamming_KnownWords_HandComputed()
    {
        Assert.Equal(64, Distances.Hamming([0UL], [ulong.MaxValue]));
        Assert.Equal(64, Distances.Hamming([0xF0F0_F0F0_F0F0_F0F0UL], [0x0F0F_0F0F_0F0F_0F0FUL]));
        Assert.Equal(4, Distances.Hamming([0b1010UL], [0b0101UL]));
        Assert.Equal(2, Distances.Hamming([0b101UL], [0b011UL]));
        Assert.Equal(0, Distances.Hamming([0xDEAD_BEEF_DEAD_BEEFUL], [0xDEAD_BEEF_DEAD_BEEFUL]));
    }

    [Fact]
    public void Hamming_SingleBits_DifferByOne()
    {
        for (int bit = 0; bit < 64; bit++)
        {
            ulong a = 0;
            ulong b = 1UL << bit;
            Assert.Equal(1, Distances.Hamming([a], [b]));
            Assert.Equal(1, BitKernels.HammingV512([a], [b]));
        }
    }

    [Fact]
    public void Hamming_MatchesScalarPopcountPerWord()
    {
        var a = DeterministicRandom.PackedBits(9, 0x5eed_3004);
        var b = DeterministicRandom.PackedBits(9, 0x5eed_3005);

        int expected = 0;
        for (int i = 0; i < a.Length; i++)
        {
            expected += BitOperations.PopCount(a[i] ^ b[i]);
        }

        Assert.Equal(expected, Distances.Hamming(a, b));
        Assert.Equal(expected, BitKernels.HammingV128(a, b));
        Assert.Equal(expected, BitKernels.HammingV256(a, b));
        Assert.Equal(expected, BitKernels.HammingV512(a, b));
    }

    [Fact]
    public void Hamming_EmptyVectors_ReturnZero()
    {
        Assert.Equal(0, Distances.Hamming([], []));
    }

    [Fact]
    public void Hamming_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => Distances.Hamming(new ulong[2], new ulong[3]));
        Assert.Throws<ArgumentException>(() => Distances.HammingScalar(new ulong[2], new ulong[3]));
    }
}
