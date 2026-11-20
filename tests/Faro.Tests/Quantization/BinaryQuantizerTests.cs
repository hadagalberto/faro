using Faro.Testing;
using Xunit;

namespace Faro.Tests.Quantization;

public class BinaryQuantizerTests
{
    [Fact]
    public void Create_PropertiesMatchDimensions()
    {
        var quantizer = BinaryQuantizer.Create(5, Distance.Cosine);

        Assert.Equal(QuantizationKind.Binary, quantizer.Kind);
        Assert.Equal(5, quantizer.Dimensions);
        Assert.Equal(Distance.Cosine, quantizer.Metric);
        Assert.Equal(8, quantizer.CodeLength); // one 64-bit word
        Assert.Equal(16, BinaryQuantizer.Create(65, Distance.Cosine).CodeLength); // two words
    }

    [Fact]
    public void Encode_KnownSigns_MatchesHandComputedBits()
    {
        var quantizer = BinaryQuantizer.Create(5, Distance.Cosine);
        var code = new byte[8];

        quantizer.Encode([1f, -2f, 3f, -4f, 5f], code);

        // Bits 1 and 3 set: 0b0000_1010 == 10.
        Assert.Equal(10UL, BitConverter.ToUInt64(code));
    }

    [Fact]
    public void Encode_NegativeZero_SetsSignBit()
    {
        var quantizer = BinaryQuantizer.Create(2, Distance.Cosine);
        var code = new byte[8];

        quantizer.Encode([-0.0f, 0.0f], code);

        Assert.Equal(1UL, BitConverter.ToUInt64(code));
    }

    [Fact]
    public void Decode_KnownBits_ReconstructsPlusMinusOne()
    {
        var quantizer = BinaryQuantizer.Create(5, Distance.Cosine);
        var code = new byte[8];
        var decoded = new float[5];

        quantizer.Encode([1f, -2f, 3f, -4f, 5f], code);
        quantizer.Decode(code, decoded);

        Assert.Equal([1f, -1f, 1f, -1f, 1f], decoded);
    }

    [Fact]
    public void EncodePacked_CopiesWordsDirectly()
    {
        var quantizer = BinaryQuantizer.Create(5, Distance.Cosine);
        var code = new byte[8];

        quantizer.EncodePacked([0b1010UL], code);

        Assert.Equal(0b1010UL, BitConverter.ToUInt64(code));
    }

    [Fact]
    public void Distance_KnownCodes_IsPopcountHamming()
    {
        var quantizer = BinaryQuantizer.Create(8, Distance.Cosine);
        var a = new byte[8];
        var b = new byte[8];

        quantizer.Encode([1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f], a); // bits 1,3,5,7
        quantizer.Encode([1f, 1f, 1f, -1f, 1f, -1f, 1f, 1f], b); // bits 3,5

        Assert.Equal(Faro.Distances.Hamming(
            [BitConverter.ToUInt64(a)], [BitConverter.ToUInt64(b)]), quantizer.Distance(a, b));
        Assert.Equal(2f, quantizer.Distance(a, b)); // bits 1 and 7 differ
    }

    [Fact]
    public void Distance_AgreesWithScalarPopcount_OnPackedRandomCodes()
    {
        const int dim = 768;
        var quantizer = BinaryQuantizer.Create(dim, Distance.Cosine);
        ulong state = 0x5eed_f001;
        var a = new byte[quantizer.CodeLength];
        var b = new byte[quantizer.CodeLength];
        var aWords = new ulong[BitVectors.WordsFor(dim)];
        var bWords = new ulong[BitVectors.WordsFor(dim)];

        for (int trial = 0; trial < 32; trial++)
        {
            for (int i = 0; i < aWords.Length; i++)
            {
                aWords[i] = DeterministicRandom.Next(ref state);
                bWords[i] = DeterministicRandom.Next(ref state);
            }

            quantizer.EncodePacked(aWords, a);
            quantizer.EncodePacked(bWords, b);

            Assert.Equal(DeterministicRandomPackedHamming(aWords, bWords), quantizer.Distance(a, b));
            Assert.Equal(Faro.Distances.HammingScalar(aWords, bWords), quantizer.Distance(a, b));
        }
    }

    [Fact]
    public void CreateAndUse_KnownInvalidInputs_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => BinaryQuantizer.Create(0, Distance.Cosine));
        Assert.Throws<ArgumentException>(() => BinaryQuantizer.Create(8, Distance.DotProduct));

        var quantizer = BinaryQuantizer.Create(5, Distance.Cosine);
        var code = new byte[8];

        Assert.Throws<ArgumentException>(() => quantizer.Encode(new[] { 1f, 2f }, code));
        Assert.Throws<ArgumentException>(() => quantizer.Encode(new[] { float.NaN }, code));
        Assert.Throws<ArgumentException>(() => quantizer.Encode([1f], Span<byte>.Empty));
        Assert.Throws<ArgumentException>(() => quantizer.EncodePacked([], code));
        Assert.Throws<ArgumentException>(() => quantizer.Decode(Span<byte>.Empty, new float[5]));
        Assert.Throws<ArgumentException>(() => quantizer.Decode(code, Span<float>.Empty));
        Assert.Throws<ArgumentException>(() => quantizer.Distance(Span<byte>.Empty, code));
    }

    private static int DeterministicRandomPackedHamming(ulong[] a, ulong[] b)
    {
        int total = 0;
        for (int i = 0; i < a.Length; i++)
        {
            ulong x = a[i] ^ b[i];
            while (x != 0)
            {
                total += (int)(x & 1);
                x >>= 1;
            }
        }

        return total;
    }
}
