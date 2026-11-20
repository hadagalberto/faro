using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Faro.Internal;

/// <summary>
/// SIMD kernels for packed-bit (binary) vector arithmetic.
/// </summary>
/// <remarks>
/// <para>
/// Bit vectors are stored packed 64 dimensions per <see langword="ulong"/> word. Hamming
/// distance is computed with a SWAR (SIMD within a register) popcount applied lane-wise —
/// the classic 64-bit popcount idiom (Warren, <c>Hacker's Delight</c> §5-1) — with explicit
/// <see cref="Vector512{T}"/>, <see cref="Vector256{T}"/>, <see cref="Vector128{T}"/> and
/// scalar paths. The scalar path uses the hardware <c>POPCNT</c> instruction via
/// <see cref="BitOperations.PopCount(ulong)"/> when available.
/// </para>
/// </remarks>
internal static class BitKernels
{
    private const ulong Step1Mask = 0x5555_5555_5555_5555UL; // 01 repeated: pairs of bits
    private const ulong Step2Mask = 0x3333_3333_3333_3333UL; // 0011 repeated: nibble halves
    private const ulong Step4Mask = 0x0F0F_0F0F_0F0F_0F0FUL; // 00001111 repeated: nibbles
    private const ulong ByteSumMultiplier = 0x0101_0101_0101_0101UL; // sums the per-byte counts into the top byte

    /// <summary>Returns the Hamming distance between two packed bit vectors using the widest hardware-accelerated vector path.</summary>
    public static int Hamming(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        if (Vector512.IsHardwareAccelerated)
        {
            return HammingV512(a, b);
        }

        if (Vector256.IsHardwareAccelerated)
        {
            return HammingV256(a, b);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            return HammingV128(a, b);
        }

        return HammingScalar(a, b);
    }

    /// <summary>Naive scalar Hamming distance (hardware POPCNT per word).</summary>
    public static int HammingScalar(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        Debug.Assert(a.Length == b.Length);

        long total = 0;
        for (int i = 0; i < a.Length; i++)
        {
            total += BitOperations.PopCount(a[i] ^ b[i]);
        }

        return (int)total;
    }

    /// <summary>Hamming distance computed with 128-bit vectors.</summary>
    public static int HammingV128(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        Debug.Assert(a.Length == b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector128<ulong>.Zero;
        ref ulong ra = ref MemoryMarshal.GetReference(a);
        ref ulong rb = ref MemoryMarshal.GetReference(b);
        var step1 = Vector128.Create(Step1Mask);
        var step2 = Vector128.Create(Step2Mask);
        var step4 = Vector128.Create(Step4Mask);
        var byteSum = Vector128.Create(ByteSumMultiplier);

        if (n >= Vector128<ulong>.Count)
        {
            for (; i <= n - Vector128<ulong>.Count; i += Vector128<ulong>.Count)
            {
                var x = Vector128.LoadUnsafe(ref ra, (nuint)i) ^ Vector128.LoadUnsafe(ref rb, (nuint)i);
                x -= (x >> 1) & step1;
                x = (x & step2) + ((x >> 2) & step2);
                x = (x + (x >> 4)) & step4;
                acc += (x * byteSum) >> 56;
            }
        }

        ulong total = 0;
        for (int e = 0; e < Vector128<ulong>.Count; e++)
        {
            total += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            // Note: `using System.Numerics` puts BigInteger's implicit (ulong, int) operators in
            // scope, which makes `ulong += int` ambiguous (CS0034); cast to keep ulong math.
            total += (ulong)BitOperations.PopCount(a[i] ^ b[i]);
        }

        return (int)total;
    }

    /// <summary>Hamming distance computed with 256-bit vectors.</summary>
    public static int HammingV256(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        Debug.Assert(a.Length == b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector256<ulong>.Zero;
        ref ulong ra = ref MemoryMarshal.GetReference(a);
        ref ulong rb = ref MemoryMarshal.GetReference(b);
        var step1 = Vector256.Create(Step1Mask);
        var step2 = Vector256.Create(Step2Mask);
        var step4 = Vector256.Create(Step4Mask);
        var byteSum = Vector256.Create(ByteSumMultiplier);

        if (n >= Vector256<ulong>.Count)
        {
            for (; i <= n - Vector256<ulong>.Count; i += Vector256<ulong>.Count)
            {
                var x = Vector256.LoadUnsafe(ref ra, (nuint)i) ^ Vector256.LoadUnsafe(ref rb, (nuint)i);
                x -= (x >> 1) & step1;
                x = (x & step2) + ((x >> 2) & step2);
                x = (x + (x >> 4)) & step4;
                acc += (x * byteSum) >> 56;
            }
        }

        ulong total = 0;
        for (int e = 0; e < Vector256<ulong>.Count; e++)
        {
            total += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            total += (ulong)BitOperations.PopCount(a[i] ^ b[i]);
        }

        return (int)total;
    }

    /// <summary>Hamming distance computed with 512-bit vectors.</summary>
    public static int HammingV512(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        Debug.Assert(a.Length == b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector512<ulong>.Zero;
        ref ulong ra = ref MemoryMarshal.GetReference(a);
        ref ulong rb = ref MemoryMarshal.GetReference(b);
        var step1 = Vector512.Create(Step1Mask);
        var step2 = Vector512.Create(Step2Mask);
        var step4 = Vector512.Create(Step4Mask);
        var byteSum = Vector512.Create(ByteSumMultiplier);

        if (n >= Vector512<ulong>.Count)
        {
            for (; i <= n - Vector512<ulong>.Count; i += Vector512<ulong>.Count)
            {
                var x = Vector512.LoadUnsafe(ref ra, (nuint)i) ^ Vector512.LoadUnsafe(ref rb, (nuint)i);
                x -= (x >> 1) & step1;
                x = (x & step2) + ((x >> 2) & step2);
                x = (x + (x >> 4)) & step4;
                acc += (x * byteSum) >> 56;
            }
        }

        ulong total = 0;
        for (int e = 0; e < Vector512<ulong>.Count; e++)
        {
            total += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            total += (ulong)BitOperations.PopCount(a[i] ^ b[i]);
        }

        return (int)total;
    }
}
