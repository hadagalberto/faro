using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Faro.Internal;

/// <summary>
/// SIMD kernels for float vector arithmetic.
/// </summary>
/// <remarks>
/// <para>
/// Every operation has four explicit code paths — <see cref="Vector512{T}"/>,
/// <see cref="Vector256{T}"/>, <see cref="Vector128{T}"/> and a scalar path — selected at
/// runtime by hardware acceleration. The scalar path accumulates in <see langword="double"/>
/// so it doubles as the high-accuracy reference implementation the public API exposes as
/// <c>*Scalar</c> methods; the equivalence test suite drives all four paths directly.
/// </para>
/// <para>
/// Vectors are processed lane-wise in chunks of the selected width, followed by a scalar
/// remainder loop for dimensions that are not a multiple of the width. The horizontal
/// reduction sums lanes in ascending lane order.
/// </para>
/// </remarks>
internal static class VectorKernels
{
    /// <summary>Returns the inner product a·b using the widest hardware-accelerated vector path.</summary>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
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

    /// <summary>Returns the naive scalar inner product, accumulated in double precision.</summary>
    public static float DotScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Debug.Assert(a.Length == b.Length);

        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += (double)a[i] * b[i];
        }

        return (float)sum;
    }

    /// <summary>Inner product computed with 128-bit vectors.</summary>
    public static float DotV128(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Debug.Assert(a.Length == b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector128<float>.Zero;
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector128<float>.Count)
        {
            for (; i <= n - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                var va = Vector128.LoadUnsafe(ref ra, (nuint)i);
                var vb = Vector128.LoadUnsafe(ref rb, (nuint)i);
                acc += va * vb;
            }
        }

        float sum = 0;
        for (int e = 0; e < Vector128<float>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>Inner product computed with 256-bit vectors.</summary>
    public static float DotV256(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Debug.Assert(a.Length == b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector256<float>.Zero;
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector256<float>.Count)
        {
            for (; i <= n - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var va = Vector256.LoadUnsafe(ref ra, (nuint)i);
                var vb = Vector256.LoadUnsafe(ref rb, (nuint)i);
                acc += va * vb;
            }
        }

        float sum = 0;
        for (int e = 0; e < Vector256<float>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>Inner product computed with 512-bit vectors.</summary>
    public static float DotV512(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Debug.Assert(a.Length == b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector512<float>.Zero;
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector512<float>.Count)
        {
            for (; i <= n - Vector512<float>.Count; i += Vector512<float>.Count)
            {
                var va = Vector512.LoadUnsafe(ref ra, (nuint)i);
                var vb = Vector512.LoadUnsafe(ref rb, (nuint)i);
                acc += va * vb;
            }
        }

        float sum = 0;
        for (int e = 0; e < Vector512<float>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>Returns the sum of squared components (squared L2 norm) using the widest hardware-accelerated vector path.</summary>
    public static float SumSquares(ReadOnlySpan<float> v)
    {
        if (Vector512.IsHardwareAccelerated)
        {
            return SumSquaresV512(v);
        }

        if (Vector256.IsHardwareAccelerated)
        {
            return SumSquaresV256(v);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            return SumSquaresV128(v);
        }

        return SumSquaresScalar(v);
    }

    /// <summary>Returns the naive scalar sum of squares, accumulated in double precision.</summary>
    public static float SumSquaresScalar(ReadOnlySpan<float> v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++)
        {
            sum += (double)v[i] * v[i];
        }

        return (float)sum;
    }

    /// <summary>Sum of squares computed with 128-bit vectors.</summary>
    public static float SumSquaresV128(ReadOnlySpan<float> v)
    {
        int n = v.Length;
        int i = 0;
        var acc = Vector128<float>.Zero;
        ref float rv = ref MemoryMarshal.GetReference(v);

        if (n >= Vector128<float>.Count)
        {
            for (; i <= n - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                var vv = Vector128.LoadUnsafe(ref rv, (nuint)i);
                acc += vv * vv;
            }
        }

        float sum = 0;
        for (int e = 0; e < Vector128<float>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            sum += v[i] * v[i];
        }

        return sum;
    }

    /// <summary>Sum of squares computed with 256-bit vectors.</summary>
    public static float SumSquaresV256(ReadOnlySpan<float> v)
    {
        int n = v.Length;
        int i = 0;
        var acc = Vector256<float>.Zero;
        ref float rv = ref MemoryMarshal.GetReference(v);

        if (n >= Vector256<float>.Count)
        {
            for (; i <= n - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var vv = Vector256.LoadUnsafe(ref rv, (nuint)i);
                acc += vv * vv;
            }
        }

        float sum = 0;
        for (int e = 0; e < Vector256<float>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            sum += v[i] * v[i];
        }

        return sum;
    }

    /// <summary>Sum of squares computed with 512-bit vectors.</summary>
    public static float SumSquaresV512(ReadOnlySpan<float> v)
    {
        int n = v.Length;
        int i = 0;
        var acc = Vector512<float>.Zero;
        ref float rv = ref MemoryMarshal.GetReference(v);

        if (n >= Vector512<float>.Count)
        {
            for (; i <= n - Vector512<float>.Count; i += Vector512<float>.Count)
            {
                var vv = Vector512.LoadUnsafe(ref rv, (nuint)i);
                acc += vv * vv;
            }
        }

        float sum = 0;
        for (int e = 0; e < Vector512<float>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            sum += v[i] * v[i];
        }

        return sum;
    }

    /// <summary>Returns the squared Euclidean (L2) distance using the widest hardware-accelerated vector path.</summary>
    public static float EuclideanSquared(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (Vector512.IsHardwareAccelerated)
        {
            return EuclideanSquaredV512(a, b);
        }

        if (Vector256.IsHardwareAccelerated)
        {
            return EuclideanSquaredV256(a, b);
        }

        if (Vector128.IsHardwareAccelerated)
        {
            return EuclideanSquaredV128(a, b);
        }

        return EuclideanSquaredScalar(a, b);
    }

    /// <summary>Returns the naive scalar squared L2 distance, accumulated in double precision.</summary>
    public static float EuclideanSquaredScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Debug.Assert(a.Length == b.Length);

        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            sum += d * d;
        }

        return (float)sum;
    }

    /// <summary>Squared L2 distance computed with 128-bit vectors.</summary>
    public static float EuclideanSquaredV128(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Debug.Assert(a.Length == b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector128<float>.Zero;
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector128<float>.Count)
        {
            for (; i <= n - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                var d = Vector128.LoadUnsafe(ref ra, (nuint)i) - Vector128.LoadUnsafe(ref rb, (nuint)i);
                acc += d * d;
            }
        }

        float sum = 0;
        for (int e = 0; e < Vector128<float>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

    /// <summary>Squared L2 distance computed with 256-bit vectors.</summary>
    public static float EuclideanSquaredV256(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Debug.Assert(a.Length == b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector256<float>.Zero;
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector256<float>.Count)
        {
            for (; i <= n - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var d = Vector256.LoadUnsafe(ref ra, (nuint)i) - Vector256.LoadUnsafe(ref rb, (nuint)i);
                acc += d * d;
            }
        }

        float sum = 0;
        for (int e = 0; e < Vector256<float>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

    /// <summary>Squared L2 distance computed with 512-bit vectors.</summary>
    public static float EuclideanSquaredV512(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Debug.Assert(a.Length == b.Length);

        int n = a.Length;
        int i = 0;
        var acc = Vector512<float>.Zero;
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref MemoryMarshal.GetReference(b);

        if (n >= Vector512<float>.Count)
        {
            for (; i <= n - Vector512<float>.Count; i += Vector512<float>.Count)
            {
                var d = Vector512.LoadUnsafe(ref ra, (nuint)i) - Vector512.LoadUnsafe(ref rb, (nuint)i);
                acc += d * d;
            }
        }

        float sum = 0;
        for (int e = 0; e < Vector512<float>.Count; e++)
        {
            sum += acc.GetElement(e);
        }

        for (; i < n; i++)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

    /// <summary>
    /// Returns the cosine distance <c>1 - a·b/(|a||b|)</c>, in [0, 2]. A vector with zero norm
    /// is treated as orthogonal to everything (distance 1).
    /// </summary>
    public static float Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float aa = SumSquares(a);
        float bb = SumSquares(b);
        if (aa == 0f || bb == 0f)
        {
            return 1f;
        }

        float cos = Dot(a, b) / MathF.Sqrt(aa * bb);
        if (cos > 1f)
        {
            cos = 1f;
        }
        else if (cos < -1f)
        {
            cos = -1f;
        }

        return 1f - cos;
    }
}
