using Faro;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.BitVectorTests;

public class BitVectorsTests
{
    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(63, 1)]
    [InlineData(64, 1)]
    [InlineData(65, 2)]
    [InlineData(128, 2)]
    [InlineData(768, 12)]
    [InlineData(769, 13)]
    public void WordsFor_CoversAllBits(int dimensions, int expected)
    {
        Assert.Equal(expected, BitVectors.WordsFor(dimensions));
    }

    [Fact]
    public void WordsFor_Negative_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => BitVectors.WordsFor(-1));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(768)]
    public void PackSignBits_SetsExactlyNegativeSignBits(int dimensions)
    {
        var vector = DeterministicRandom.UniformVector(dimensions, 0x5eed_5000 + (ulong)dimensions);
        ulong[] packed = BitVectors.PackSignBits(vector);

        Assert.Equal(BitVectors.WordsFor(dimensions), packed.Length);
        for (int i = 0; i < dimensions; i++)
        {
            bool bitSet = (packed[i >> 6] & (1UL << (i & 63))) != 0;
            Assert.Equal(vector[i] < 0, bitSet);
        }
    }

    [Fact]
    public void PackSignBits_NegativeZeroAndSignedNaNSetTheirBit()
    {
        var vector = new float[]
        {
            0f,              // +0 -> bit 0
            -0f,             // -0 -> bit 1 (raw sign bit)
            float.NaN,       // positive NaN -> bit 2 clear
            NegativeNaN(),   // negative NaN -> bit 3 set
            -1f,             // -> bit 4 set
            1f,              // -> bit 5 clear
        };

        ulong packed = BitVectors.PackSignBits(vector)[0];
        Assert.Equal(0b0001_1010UL, packed);

        static float NegativeNaN() => BitConverter.Int32BitsToSingle(unchecked((int)0xFFC0_0000));
    }

    [Fact]
    public void PackSignBits_PartialLastWord_HasZeroTailBits()
    {
        // 65 dimensions: the last word carries exactly one dimension (bit 0); bits 1..63 stay 0.
        var vector = new float[65];
        vector[0] = -1f;   // word 0, bit 0
        vector[64] = -1f;  // word 1, bit 0

        ulong[] packed = BitVectors.PackSignBits(vector);
        Assert.Equal(2, packed.Length);
        Assert.Equal(1UL, packed[0]);
        Assert.Equal(1UL, packed[1]);
    }

    [Fact]
    public void PackUnpack_RoundTripsSigns()
    {
        var vector = DeterministicRandom.UniformVector(200, 0x5eed_5001);
        ulong[] packed = BitVectors.PackSignBits(vector);

        var unpacked = new float[200];
        BitVectors.UnpackSignBits(packed, 200, unpacked);

        for (int i = 0; i < 200; i++)
        {
            Assert.Equal(vector[i] < 0 ? -1f : 1f, unpacked[i]);
        }
    }

    [Fact]
    public void PackSignBits_IntoLargerDestination_OnlyTouchesNeededWords()
    {
        var vector = new float[] { -1f, 1f, -1f };
        var destination = new ulong[] { 0xAAAA_AAAA_AAAA_AAAAUL, 0x5555_5555_5555_5555UL };

        BitVectors.PackSignBits(vector, destination);

        Assert.Equal(0b101UL, destination[0]);
        Assert.Equal(0x5555_5555_5555_5555UL, destination[1]);
    }

    [Fact]
    public void PackSignBits_DestinationTooSmall_Throws()
    {
        Assert.Throws<ArgumentException>(() => BitVectors.PackSignBits(new float[65], new ulong[1]));
    }

    [Fact]
    public void UnpackSignBits_DestinationTooSmall_Throws()
    {
        Assert.Throws<ArgumentException>(() => BitVectors.UnpackSignBits(new ulong[2], 100, new float[64]));
    }

    [Fact]
    public void UnpackSignBits_PackedTooSmall_Throws()
    {
        Assert.Throws<ArgumentException>(() => BitVectors.UnpackSignBits(new ulong[1], 65, new float[65]));
    }

    [Fact]
    public void Hamming_OverPackedForm_AgreesWithPerDimensionComparison()
    {
        var a = DeterministicRandom.UniformVector(257, 0x5eed_5002);
        var b = DeterministicRandom.UniformVector(257, 0x5eed_5003);

        int expected = 0;
        for (int i = 0; i < a.Length; i++)
        {
            if ((a[i] < 0) != (b[i] < 0))
            {
                expected++;
            }
        }

        Assert.Equal(expected, Distances.Hamming(BitVectors.PackSignBits(a), BitVectors.PackSignBits(b)));
    }
}
