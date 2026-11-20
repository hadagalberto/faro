using System.Buffers;
using System.Diagnostics;
using Faro.Internal;

namespace Faro;

/// <summary>
/// An exact (brute-force) vector index: every search scores all stored vectors with SIMD
/// distance kernels and keeps the k best with a pooled heap. This is the ground truth the
/// approximate indexes and recall suites are validated against.
/// </summary>
/// <remarks>
/// <para>
/// Searches are deterministic regardless of metric, heap contents or CPU: results are ordered
/// best-first with ties broken by ascending <see cref="SearchResult.Id"/>. Collections with at
/// least <see cref="FlatOptions.ParallelThreshold"/> vectors are searched in parallel with
/// per-partition heaps that are merged deterministically, yielding results identical to the
/// sequential path.
/// </para>
/// <para>
/// Concurrency: <see cref="Add"/> supports a single writer; searches may run concurrently with
/// each other and with a single adding writer. The count is published with a volatile write
/// after each vector is fully written, and readers clamp the visible count to the capacity of
/// the storage array they observe, so every search sees a consistent prefix of fully written
/// vectors.
/// </para>
/// <para>
/// The sequential search path is allocation-free in steady state: its heap and (for Hamming)
/// the packed query buffer are rented from <see cref="ArrayPool{T}"/> and returned afterwards;
/// only the result array of the <see cref="Search(ReadOnlySpan{float}, int)"/> overload is
/// allocated per call. The parallel path additionally rents a query copy and partition staging
/// buffers, since <see cref="Parallel"/> lambdas cannot capture spans.
/// </para>
/// </remarks>
public sealed class FlatIndex
{
    private readonly int _dimensions;
    private readonly Distance _distance;
    private readonly bool _normalize;
    private readonly int _wordsPerVector;
    private readonly int _elementsPerVector;
    private readonly int _parallelThreshold;

    private float[]? _vectors;
    private ulong[]? _bits;
    private int _capacity;
    private int _count;

    /// <summary>Creates an exact index with the given options.</summary>
    /// <param name="options">The index configuration.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <see cref="FlatOptions.Dimensions"/> or <see cref="FlatOptions.InitialCapacity"/> is not positive,
    /// or <see cref="FlatOptions.ParallelThreshold"/> is negative, or the initial capacity exceeds the maximum
    /// representable array size.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// <see cref="FlatOptions.NormalizeOnAdd"/> is <see langword="true"/> with a distance other than
    /// <see cref="Distance.Cosine"/>.
    /// </exception>
    public FlatIndex(FlatOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        if (options.Dimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.Dimensions, "Dimensions must be positive.");
        }

        if (options.InitialCapacity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.InitialCapacity, "InitialCapacity must be positive.");
        }

        if (options.ParallelThreshold < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.ParallelThreshold, "ParallelThreshold must not be negative.");
        }

        _dimensions = options.Dimensions;
        _distance = options.Distance;
        _normalize = options.NormalizeOnAdd ?? (options.Distance == Distance.Cosine);
        if (_normalize && _distance != Distance.Cosine)
        {
            throw new ArgumentException("NormalizeOnAdd is only supported for Distance.Cosine.", nameof(options));
        }

        _wordsPerVector = BitVectors.WordsFor(_dimensions);
        _elementsPerVector = _distance == Distance.Hamming ? _wordsPerVector : _dimensions;
        _parallelThreshold = options.ParallelThreshold;

        _capacity = options.InitialCapacity;
        if ((long)_capacity * _elementsPerVector > Array.MaxLength)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.InitialCapacity, "InitialCapacity exceeds the maximum array size for these dimensions.");
        }

        if (_distance == Distance.Hamming)
        {
            _bits = new ulong[_capacity * _wordsPerVector];
        }
        else
        {
            _vectors = new float[_capacity * _dimensions];
        }
    }

    /// <summary>Number of vectors currently stored.</summary>
    public int Count => Volatile.Read(ref _count);

    /// <summary>Number of dimensions of stored vectors.</summary>
    public int Dimensions => _dimensions;

    /// <summary>The distance metric of this index.</summary>
    public Distance Distance => _distance;

    /// <summary>
    /// Appends a vector and returns its internal identifier (sequential from 0). For
    /// <see cref="Distance.Cosine"/> with normalization on (the default), the stored vector is
    /// normalized to unit length; for <see cref="Distance.Hamming"/> it is sign-bit packed.
    /// </summary>
    /// <param name="vector">The vector to store; its length must equal <see cref="Dimensions"/>.</param>
    /// <returns>The internal identifier of the stored vector.</returns>
    /// <exception cref="ArgumentException"><paramref name="vector"/> has the wrong length or contains non-finite values.</exception>
    /// <remarks>Supports a single concurrent writer; searches may run concurrently with adds.</remarks>
    public int Add(ReadOnlySpan<float> vector)
    {
        ValidateVector(vector);

        EnsureCapacity(_count + 1);
        int id = _count;
        if (_distance == Distance.Hamming)
        {
            BitVectors.PackSignBits(vector, _bits!.AsSpan(id * _wordsPerVector, _wordsPerVector));
        }
        else if (_normalize)
        {
            Distances.CopyNormalized(vector, _vectors!.AsSpan(id * _dimensions, _dimensions));
        }
        else
        {
            vector.CopyTo(_vectors!.AsSpan(id * _dimensions, _dimensions));
        }

        Volatile.Write(ref _count, id + 1);
        return id;
    }

    /// <summary>
    /// Searches for the <paramref name="k"/> nearest vectors to <paramref name="query"/>, best
    /// first, and returns them in a freshly allocated array of at most
    /// min(k, <see cref="Count"/>) hits.
    /// </summary>
    /// <param name="query">The query vector; its length must equal <see cref="Dimensions"/>.</param>
    /// <param name="k">Maximum number of hits to return; must be positive.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="k"/> is not positive.</exception>
    /// <exception cref="ArgumentException"><paramref name="query"/> has the wrong length or contains non-finite values.</exception>
    public SearchResult[] Search(ReadOnlySpan<float> query, int k)
    {
        ValidateQuery(query);
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), k, "k must be positive.");
        }

        int n = SnapshotCount();
        int take = Math.Min(k, n);
        if (take == 0)
        {
            return [];
        }

        var results = new SearchResult[take];
        SearchCore(query, take, results, n);
        return results;
    }

    /// <summary>
    /// Searches for the nearest vectors, best first, writing into <paramref name="results"/> and
    /// returning the number of hits written (at most min(k, <see cref="Count"/>, results.Length)).
    /// This overload performs no allocations in steady state.
    /// </summary>
    /// <param name="query">The query vector; its length must equal <see cref="Dimensions"/>.</param>
    /// <param name="k">Maximum number of hits to return; must be positive.</param>
    /// <param name="results">The destination buffer for the hits.</param>
    /// <returns>The number of hits written to <paramref name="results"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="k"/> is not positive.</exception>
    /// <exception cref="ArgumentException"><paramref name="query"/> has the wrong length or contains non-finite values.</exception>
    public int Search(ReadOnlySpan<float> query, int k, Span<SearchResult> results)
    {
        ValidateQuery(query);
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), k, "k must be positive.");
        }

        if (results.Length == 0)
        {
            return 0;
        }

        int n = SnapshotCount();
        if (n == 0)
        {
            return 0;
        }

        int limit = Math.Min(k, results.Length);
        SearchCore(query, limit, results, n);
        return Math.Min(limit, n);
    }

    /// <summary>
    /// Copies the stored form of the vector with the given id into <paramref name="destination"/>
    /// (normalized for cosine-with-normalization, sign-reconstructed ±1 values for Hamming).
    /// </summary>
    /// <param name="id">The internal vector identifier.</param>
    /// <param name="destination">The destination span; must have <see cref="Dimensions"/> entries.</param>
    /// <returns><see langword="false"/> if <paramref name="id"/> is beyond the stored vectors.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="id"/> is negative.</exception>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has the wrong length.</exception>
    public bool TryGetVector(int id, Span<float> destination)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(id);
        if (destination.Length != _dimensions)
        {
            throw new ArgumentException($"destination must have {_dimensions} entries, got {destination.Length}.", nameof(destination));
        }

        int n = SnapshotCount();
        if (id >= n)
        {
            return false;
        }

        if (_distance == Distance.Hamming)
        {
            BitVectors.UnpackSignBits(_bits!.AsSpan(id * _wordsPerVector, _wordsPerVector), _dimensions, destination);
        }
        else
        {
            _vectors!.AsSpan(id * _dimensions, _dimensions).CopyTo(destination);
        }

        return true;
    }

    private void SearchCore(ReadOnlySpan<float> query, int limit, Span<SearchResult> results, int n)
    {
        if (n >= _parallelThreshold && n >= 2)
        {
            SearchParallel(query, limit, results, n);
        }
        else
        {
            SearchSequential(query, limit, results, n);
        }
    }

    private static HeapOrder TopKOrder(Distance distance) =>
        Distances.LowerIsCloser(distance) ? HeapOrder.MaxIdDescending : HeapOrder.MinIdDescending;

    private void SearchSequential(ReadOnlySpan<float> query, int limit, Span<SearchResult> results, int n)
    {
        using var heap = DistanceHeap.Rent(TopKOrder(_distance), limit);

        // Normalized-cosine storage: normalize the query into a pooled buffer once so scores
        // are true cosine distances (1 - dot of two unit vectors) while keeping the dot fast path.
        float[]? normalizedQuery = null;
        ReadOnlySpan<float> effectiveQuery = query;
        if (_normalize)
        {
            normalizedQuery = ArrayPool<float>.Shared.Rent(_dimensions);
            // ArrayPool returns an array at least as large as requested; CopyNormalized requires
            // an exact-length destination, so slice before handing it over.
            Distances.CopyNormalized(query, normalizedQuery.AsSpan(0, _dimensions));
            effectiveQuery = normalizedQuery.AsSpan(0, _dimensions);
        }

        ulong[]? packedQuery = null;
        if (_distance == Distance.Hamming)
        {
            packedQuery = ArrayPool<ulong>.Shared.Rent(_wordsPerVector);
            BitVectors.PackSignBits(query, packedQuery);
        }

        try
        {
            var vectors = _vectors;
            var bits = _bits;
            for (int id = 0; id < n; id++)
            {
                float score = ScoreVector(effectiveQuery, packedQuery, vectors, bits, id);
                heap.PushBounded(limit, score, id);
            }

            DrainBestFirst(heap, results);
        }
        finally
        {
            if (normalizedQuery is not null)
            {
                ArrayPool<float>.Shared.Return(normalizedQuery);
            }

            if (packedQuery is not null)
            {
                ArrayPool<ulong>.Shared.Return(packedQuery);
            }
        }
    }

    private void SearchParallel(ReadOnlySpan<float> query, int limit, Span<SearchResult> results, int n)
    {
        // Parallel bodies are lambdas, which cannot capture spans: copy the query into a
        // rented managed array (normalizing it first for normalized-cosine storage) and let
        // the bodies slice it. All pooled scratch is returned after Parallel.For joins.
        var queryBuffer = ArrayPool<float>.Shared.Rent(_dimensions);
        if (_normalize)
        {
            Distances.CopyNormalized(query, queryBuffer.AsSpan(0, _dimensions));
        }
        else
        {
            query.CopyTo(queryBuffer);
        }

        ulong[]? packedQuery = null;
        if (_distance == Distance.Hamming)
        {
            packedQuery = ArrayPool<ulong>.Shared.Rent(_wordsPerVector);
            BitVectors.PackSignBits(query, packedQuery);
        }

        int partitions = Math.Min(n, Environment.ProcessorCount);
        int chunk = (n + partitions - 1) / partitions;
        partitions = (n + chunk - 1) / chunk;

        var staging = ArrayPool<SearchResult>.Shared.Rent(partitions * limit);
        var counts = ArrayPool<int>.Shared.Rent(partitions);
        var topKOrder = TopKOrder(_distance);
        var vectors = _vectors;
        var bits = _bits;
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = partitions };

        try
        {
            Parallel.For(0, partitions, parallelOptions, partition =>
            {
                int begin = partition * chunk;
                int end = Math.Min(n, begin + chunk);

                using var heap = DistanceHeap.Rent(topKOrder, limit);
                for (int id = begin; id < end; id++)
                {
                    float score = ScoreVector(queryBuffer.AsSpan(0, _dimensions), packedQuery, vectors, bits, id);
                    heap.PushBounded(limit, score, id);
                }

                counts[partition] = heap.Count;
                int index = partition * limit + heap.Count;
                while (heap.TryPop(out float score, out int id))
                {
                    staging[--index] = new SearchResult(id, score);
                }
            });

            using var finalHeap = DistanceHeap.Rent(topKOrder, limit);
            for (int partition = 0; partition < partitions; partition++)
            {
                int found = counts[partition];
                var segment = staging.AsSpan(partition * limit, found);
                for (int i = 0; i < found; i++)
                {
                    finalHeap.PushBounded(limit, segment[i].Score, segment[i].Id);
                }
            }

            DrainBestFirst(finalHeap, results);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(queryBuffer);
            if (packedQuery is not null)
            {
                ArrayPool<ulong>.Shared.Return(packedQuery);
            }

            ArrayPool<SearchResult>.Shared.Return(staging);
            ArrayPool<int>.Shared.Return(counts);
        }
    }

    private float ScoreVector(
        ReadOnlySpan<float> query,
        ulong[]? packedQuery,
        float[]? vectors,
        ulong[]? bits,
        int id)
    {
        switch (_distance)
        {
            case Distance.Cosine:
            {
                Debug.Assert(vectors is not null);
                ReadOnlySpan<float> vector = vectors.AsSpan(id * _dimensions, _dimensions);
                // Both operands are unit vectors here (storage is normalized on add, and the
                // query is normalized into a pooled buffer per search), so the cosine distance
                // is exactly 1 - dot — the cheapest fully-correct path.
                return _normalize
                    ? 1f - VectorKernels.Dot(query, vector)
                    : VectorKernels.Cosine(query, vector);
            }

            case Distance.DotProduct:
                Debug.Assert(vectors is not null);
                return VectorKernels.Dot(query, vectors.AsSpan(id * _dimensions, _dimensions));

            case Distance.Euclidean:
                Debug.Assert(vectors is not null);
                return VectorKernels.EuclideanSquared(query, vectors.AsSpan(id * _dimensions, _dimensions));

            case Distance.Hamming:
                Debug.Assert(packedQuery is not null && bits is not null);
                return BitKernels.Hamming(
                    packedQuery.AsSpan(0, _wordsPerVector),
                    bits.AsSpan(id * _wordsPerVector, _wordsPerVector));

            default:
                throw new InvalidOperationException($"Unknown distance metric '{_distance}'.");
        }
    }

    private static void DrainBestFirst(DistanceHeap heap, Span<SearchResult> results)
    {
        // The heap root is always the worst kept item, so popping yields worst-first; writing
        // backwards produces the best-first order the API promises. With ties broken by id the
        // order is a total order, hence deterministic.
        int index = heap.Count;
        while (heap.TryPop(out float score, out int id))
        {
            results[--index] = new SearchResult(id, score);
        }
    }

    private int SnapshotCount()
    {
        // A single writer publishes: (data written) -> volatile count. Readers pair a volatile
        // count read with the storage array they observe and clamp, so they only ever score ids
        // whose data is fully visible in that array. Arrays never shrink, so any array observed
        // later holds a superset of the data of the array used for clamping.
        int count = Volatile.Read(ref _count);
        if (_distance == Distance.Hamming)
        {
            return Math.Min(count, _bits!.Length / _wordsPerVector);
        }

        return Math.Min(count, _vectors!.Length / _dimensions);
    }

    private void EnsureCapacity(int required)
    {
        if (required <= _capacity)
        {
            return;
        }

        int newCapacity = Math.Max(required, (int)Math.Min((long)_capacity * 2, int.MaxValue));
        if ((long)newCapacity * _elementsPerVector > Array.MaxLength)
        {
            newCapacity = (int)(Array.MaxLength / _elementsPerVector);
            if (newCapacity < required)
            {
                throw new OutOfMemoryException($"FlatIndex cannot grow beyond {newCapacity} vectors for {_elementsPerVector} elements per vector.");
            }
        }

        if (_distance == Distance.Hamming)
        {
            var grown = new ulong[newCapacity * _wordsPerVector];
            Array.Copy(_bits!, grown, (long)_count * _wordsPerVector);
            _bits = grown;
        }
        else
        {
            var grown = new float[newCapacity * _dimensions];
            Array.Copy(_vectors!, grown, (long)_count * _dimensions);
            _vectors = grown;
        }

        _capacity = newCapacity;
    }

    private void ValidateVector(ReadOnlySpan<float> vector)
    {
        if (vector.Length != _dimensions)
        {
            throw new ArgumentException($"vector must have {_dimensions} dimensions, got {vector.Length}.", nameof(vector));
        }

        for (int i = 0; i < vector.Length; i++)
        {
            if (!float.IsFinite(vector[i]))
            {
                throw new ArgumentException($"vector contains a non-finite value at index {i}.", nameof(vector));
            }
        }
    }

    private void ValidateQuery(ReadOnlySpan<float> query)
    {
        if (query.Length != _dimensions)
        {
            throw new ArgumentException($"query must have {_dimensions} dimensions, got {query.Length}.", nameof(query));
        }

        for (int i = 0; i < query.Length; i++)
        {
            if (!float.IsFinite(query[i]))
            {
                throw new ArgumentException($"query contains a non-finite value at index {i}.", nameof(query));
            }
        }
    }
}
