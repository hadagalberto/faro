using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Faro.Internal;

/// <summary>
/// SIMD kernels for signed 8-bit (int8) quantized vectors.
/// </summary>
/// <remarks>
/// <para>
/// Codes are stored as <see langword="sbyte"/> in [-127, 127] (symmetric scalar quantization).
/// Products of two codes fit in a 16-bit integer (127*127 = 16129), so each block is widened
/// sbyte -> short -> int and multiplied in 32-bit lanes. This avoids per-code dequantization
/// and keeps the hot loop in integer SIMD registers.
/// </para>
/// <para>
/// Every operation has four paths — <see cref="Vector512{T}"/>, <see cref="Vector256{T}"/>,
/// <see cref="Vector128{T}"/> and scalar — selected at runtime. The scalar twins use
/// <see langword="int"/> accumulation and double as exact references for the equivalence tests.
/// Remainders after the last full block are handled by a scalar tail loop, so every path returns
/// the identical integer result.
/// </para>
/// </remarks>
internal static class Int8Kernels
{
    /// <summary>Returns the inner product of two int8 codes using the widest hardware-accelerated path.</summary>
    public static int Dot(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        if (Vector512.IsHardwareAccelerated)
        {
            return DotV512(a, b);
        }

        if (Vector256.IsHardwareAccelerated)
        {
            return DotV256(a, b);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            return DotV128(a, b);
        }

        return DotScalar(a, b);
    }

    /// <summary>Returns the naive scalar inner product of two int8 codes (reference implementation).</summary>
    public static int DotScalar(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        RequireSameLength(a.Length, b.Length);

        int sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>Inner product of two int8 codes computed with 128-bit vectors.</summary>
    public static int DotV128(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        RequireSameLength(a.Length, b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector128<int>.Zero;
        ref sbyte ra = ref MemoryMarshal.GetReference(a);
        ref sbyte rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector128<sbyte>.Count)
        {
            for (; i <= n - Vector128<sbyte>.Count; i += Vector128<sbyte>.Count)
            {
                acc += Multiply128(
                    Vector128.LoadUnsafe(ref ra, (nuint)i),
                    Vector128.LoadUnsafe(ref rb, (nuint)i));
            }
        }

        int sum = 0;
        for (int e = 0; e < Vector128<int>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>Inner product of two int8 codes computed with 256-bit vectors.</summary>
    public static int DotV256(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        RequireSameLength(a.Length, b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector256<int>.Zero;
        ref sbyte ra = ref MemoryMarshal.GetReference(a);
        ref sbyte rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector256<sbyte>.Count)
        {
            for (; i <= n - Vector256<sbyte>.Count; i += Vector256<sbyte>.Count)
            {
                acc += Multiply256(
                    Vector256.LoadUnsafe(ref ra, (nuint)i),
                    Vector256.LoadUnsafe(ref rb, (nuint)i));
            }
        }

        int sum = 0;
        for (int e = 0; e < Vector256<int>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>Inner product of two int8 codes computed with 512-bit vectors.</summary>
    public static int DotV512(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        RequireSameLength(a.Length, b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector512<int>.Zero;
        ref sbyte ra = ref MemoryMarshal.GetReference(a);
        ref sbyte rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector512<sbyte>.Count)
        {
            for (; i <= n - Vector512<sbyte>.Count; i += Vector512<sbyte>.Count)
            {
                acc += Multiply512(
                    Vector512.LoadUnsafe(ref ra, (nuint)i),
                    Vector512.LoadUnsafe(ref rb, (nuint)i));
            }
        }

        int sum = 0;
        for (int e = 0; e < Vector512<int>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>Returns the squared Euclidean distance between two int8 codes (reference implementation).</summary>
    public static int SquaredDistanceScalar(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        RequireSameLength(a.Length, b.Length);

        int sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            int d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

    /// <summary>Squared Euclidean distance between two int8 codes using the widest accelerated path.</summary>
    public static int SquaredDistance(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        if (Vector512.IsHardwareAccelerated)
        {
            return SquaredDistanceV512(a, b);
        }

        if (Vector256.IsHardwareAccelerated)
        {
            return SquaredDistanceV256(a, b);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            return SquaredDistanceV128(a, b);
        }

        return SquaredDistanceScalar(a, b);
    }

    /// <summary>Squared Euclidean distance between two int8 codes computed with 128-bit vectors.</summary>
    public static int SquaredDistanceV128(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        RequireSameLength(a.Length, b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector128<int>.Zero;
        ref sbyte ra = ref MemoryMarshal.GetReference(a);
        ref sbyte rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector128<sbyte>.Count)
        {
            for (; i <= n - Vector128<sbyte>.Count; i += Vector128<sbyte>.Count)
            {
                acc += SquareDifference128(
                    Vector128.LoadUnsafe(ref ra, (nuint)i),
                    Vector128.LoadUnsafe(ref rb, (nuint)i));
            }
        }

        int sum = 0;
        for (int e = 0; e < Vector128<int>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            int d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

    /// <summary>Squared Euclidean distance between two int8 codes computed with 256-bit vectors.</summary>
    public static int SquaredDistanceV256(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        RequireSameLength(a.Length, b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector256<int>.Zero;
        ref sbyte ra = ref MemoryMarshal.GetReference(a);
        ref sbyte rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector256<sbyte>.Count)
        {
            for (; i <= n - Vector256<sbyte>.Count; i += Vector256<sbyte>.Count)
            {
                acc += SquareDifference256(
                    Vector256.LoadUnsafe(ref ra, (nuint)i),
                    Vector256.LoadUnsafe(ref rb, (nuint)i));
            }
        }

        int sum = 0;
        for (int e = 0; e < Vector256<int>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            int d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

    /// <summary>Squared Euclidean distance between two int8 codes computed with 512-bit vectors.</summary>
    public static int SquaredDistanceV512(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        RequireSameLength(a.Length, b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector512<int>.Zero;
        ref sbyte ra = ref MemoryMarshal.GetReference(a);
        ref sbyte rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector512<sbyte>.Count)
        {
            for (; i <= n - Vector512<sbyte>.Count; i += Vector512<sbyte>.Count)
            {
                acc += SquareDifference512(
                    Vector512.LoadUnsafe(ref ra, (nuint)i),
                    Vector512.LoadUnsafe(ref rb, (nuint)i));
            }
        }

        int sum = 0;
        for (int e = 0; e < Vector512<int>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            int d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

    // LoadUnsafe reads a whole SIMD register from a raw pointer; a short operand is an out-of-bounds
    // read once Debug.Assert is stripped, so all kernels validate their lengths in Release too.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RequireSameLength(int left, int right)
    {
        if (left != right)
        {
            ThrowLengthMismatch(left, right);
        }
    }

    [DoesNotReturn]
    private static void ThrowLengthMismatch(int left, int right) =>
        throw new ArgumentException($"int8 codes must have the same length, got {left} and {right}.");

    private static Vector128<int> Multiply128(Vector128<sbyte> a, Vector128<sbyte> b)
    {
        var al = Vector128.WidenLower(a);
        var bl = Vector128.WidenLower(b);
        var ah = Vector128.WidenUpper(a);
        var bh = Vector128.WidenUpper(b);
        return Vector128.WidenLower(al) * Vector128.WidenLower(bl)
             + Vector128.WidenUpper(al) * Vector128.WidenUpper(bl)
             + Vector128.WidenLower(ah) * Vector128.WidenLower(bh)
             + Vector128.WidenUpper(ah) * Vector128.WidenUpper(bh);
    }

    private static Vector256<int> Multiply256(Vector256<sbyte> a, Vector256<sbyte> b)
    {
        var al = Vector256.WidenLower(a);
        var bl = Vector256.WidenLower(b);
        var ah = Vector256.WidenUpper(a);
        var bh = Vector256.WidenUpper(b);
        return Vector256.WidenLower(al) * Vector256.WidenLower(bl)
             + Vector256.WidenUpper(al) * Vector256.WidenUpper(bl)
             + Vector256.WidenLower(ah) * Vector256.WidenLower(bh)
             + Vector256.WidenUpper(ah) * Vector256.WidenUpper(bh);
    }

    private static Vector512<int> Multiply512(Vector512<sbyte> a, Vector512<sbyte> b)
    {
        var al = Vector512.WidenLower(a);
        var bl = Vector512.WidenLower(b);
        var ah = Vector512.WidenUpper(a);
        var bh = Vector512.WidenUpper(b);
        return Vector512.WidenLower(al) * Vector512.WidenLower(bl)
             + Vector512.WidenUpper(al) * Vector512.WidenUpper(bl)
             + Vector512.WidenLower(ah) * Vector512.WidenLower(bh)
             + Vector512.WidenUpper(ah) * Vector512.WidenUpper(bh);
    }

    private static Vector128<int> SquareDifference128(Vector128<sbyte> a, Vector128<sbyte> b)
    {
        var dl = Vector128.WidenLower(a) - Vector128.WidenLower(b);
        var dh = Vector128.WidenUpper(a) - Vector128.WidenUpper(b);
        return Vector128.WidenLower(dl) * Vector128.WidenLower(dl)
             + Vector128.WidenUpper(dl) * Vector128.WidenUpper(dl)
             + Vector128.WidenLower(dh) * Vector128.WidenLower(dh)
             + Vector128.WidenUpper(dh) * Vector128.WidenUpper(dh);
    }

    private static Vector256<int> SquareDifference256(Vector256<sbyte> a, Vector256<sbyte> b)
    {
        var dl = Vector256.WidenLower(a) - Vector256.WidenLower(b);
        var dh = Vector256.WidenUpper(a) - Vector256.WidenUpper(b);
        return Vector256.WidenLower(dl) * Vector256.WidenLower(dl)
             + Vector256.WidenUpper(dl) * Vector256.WidenUpper(dl)
             + Vector256.WidenLower(dh) * Vector256.WidenLower(dh)
             + Vector256.WidenUpper(dh) * Vector256.WidenUpper(dh);
    }

    private static Vector512<int> SquareDifference512(Vector512<sbyte> a, Vector512<sbyte> b)
    {
        var dl = Vector512.WidenLower(a) - Vector512.WidenLower(b);
        var dh = Vector512.WidenUpper(a) - Vector512.WidenUpper(b);
        return Vector512.WidenLower(dl) * Vector512.WidenLower(dl)
             + Vector512.WidenUpper(dl) * Vector512.WidenUpper(dl)
             + Vector512.WidenLower(dh) * Vector512.WidenLower(dh)
             + Vector512.WidenUpper(dh) * Vector512.WidenUpper(dh);
    }
}
