using System.Buffers;

namespace Faro.Internal;

/// <summary>
/// The total order a <see cref="DistanceHeap"/> maintains over (priority, id) pairs. The heap
/// root is always the maximum of this order.
/// </summary>
/// <remarks>
/// <para>
/// Beam-search heaps (best-first expansion) use the <c>IdAscending</c> orders: the root is the
/// best candidate per the metric, ties expanding the lower id first.
/// </para>
/// <para>
/// Top-k selection heaps use the <c>IdDescending</c> orders — the exact <em>inverse</em> of the
/// metric order, ties included — so the root is precisely the <em>worst</em> kept item and
/// <see cref="DistanceHeap.PushBounded"/> evicts exactly the item the metric would evict.
/// Renting the inverse order (rather than just inverting the priority comparison) keeps the
/// kept set exact when equal priorities straddle the k boundary: the heap root holds the
/// highest id among the worst-priority ties, so lower ids survive.
/// </para>
/// </remarks>
internal enum HeapOrder
{
    /// <summary>Root: smallest priority, ties to the lowest id. Best-first for lower-is-closer metrics (Cosine, Euclidean, Hamming).</summary>
    MinIdAscending,

    /// <summary>Root: largest priority, ties to the lowest id. Best-first for higher-is-closer metrics (DotProduct).</summary>
    MaxIdAscending,

    /// <summary>Root: largest priority, ties to the highest id. Top-k eviction heap for lower-is-closer metrics.</summary>
    MaxIdDescending,

    /// <summary>Root: smallest priority, ties to the highest id. Top-k eviction heap for higher-is-closer metrics.</summary>
    MinIdDescending,
}

/// <summary>
/// A pooled binary heap of (priority, id) pairs used by every index search to keep the k best
/// candidates without per-search allocations.
/// </summary>
/// <remarks>
/// <para>
/// The backing array is rented from <see cref="ArrayPool{T}"/> and returned on
/// <see cref="Dispose"/>; grow operations re-rent. The heap orders pairs by priority with a
/// deterministic tie-break on id (see <see cref="HeapOrder"/>), so both the kept set and the
/// pop order are deterministic. Classic array binary heap: the children of <c>i</c> are
/// <c>2i+1</c> and <c>2i+2</c>.
/// </para>
/// </remarks>
internal ref struct DistanceHeap
{
    private (float Priority, int Id)[] _items;
    private int _count;
    private readonly HeapOrder _order;

    private DistanceHeap(HeapOrder order, (float Priority, int Id)[] items)
    {
        _order = order;
        _items = items;
        _count = 0;
    }

    /// <summary>Number of items currently held.</summary>
    public int Count => _count;

    /// <summary>Capacity of the currently rented backing array.</summary>
    public int Capacity => _items.Length;

    /// <summary>Rents a heap backed by a pooled array of at least <paramref name="capacity"/> items.</summary>
    public static DistanceHeap Rent(HeapOrder order, int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        var items = ArrayPool<(float, int)>.Shared.Rent(capacity);
        return new DistanceHeap(order, items);
    }

    /// <summary>Inserts an item, growing the backing array as needed.</summary>
    public void Push(float priority, int id)
    {
        if (_count == _items.Length)
        {
            Grow();
        }

        int i = _count++;
        while (i > 0)
        {
            int parent = (i - 1) >> 1;
            if (!IsBetter((priority, id), _items[parent]))
            {
                break;
            }

            _items[i] = _items[parent];
            i = parent;
        }

        _items[i] = (priority, id);
    }

    /// <summary>
    /// Top-k insertion for eviction heaps (the <c>IdDescending</c> orders, whose root is the
    /// worst kept item per the metric): while fewer than <paramref name="limit"/> items are
    /// held, inserts; afterwards the item replaces the root only if the metric ranks it above
    /// the current worst, otherwise it is dropped.
    /// </summary>
    /// <returns><see langword="true"/> if the item is held by the heap afterwards.</returns>
    public bool PushBounded(int limit, float priority, int id)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(limit, 1);
        if (_count < limit)
        {
            Push(priority, id);
            return true;
        }

        // The root is the metric-worst kept item; the candidate is admitted exactly when the
        // root beats it in the heap order (the inverse of the metric order).
        if (IsBetter(_items[0], (priority, id)))
        {
            SiftDown(0, (priority, id), _count);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Returns whether a candidate ranks strictly above the current root in this heap's total
    /// order (see <see cref="HeapOrder"/>). An empty heap has no root, so this is always
    /// <see langword="false"/> for one.
    /// </summary>
    public bool IsBetterThanRoot(float priority, int id) =>
        _count != 0 && IsBetter((priority, id), _items[0]);

    /// <summary>Peeks the root without removing it.</summary>
    public bool TryPeekRoot(out float priority, out int id)
    {
        if (_count == 0)
        {
            priority = 0;
            id = 0;
            return false;
        }

        (priority, id) = _items[0];
        return true;
    }

    /// <summary>Removes and returns the root (the extreme item per the heap order).</summary>
    public bool TryPop(out float priority, out int id)
    {
        if (_count == 0)
        {
            priority = 0;
            id = 0;
            return false;
        }

        (priority, id) = _items[0];
        int n = --_count;
        if (n > 0)
        {
            SiftDown(0, _items[n], n);
        }

        return true;
    }

    /// <summary>Drops all items while keeping the rented backing array for reuse.</summary>
    public void Clear() => _count = 0;

    /// <summary>Returns the backing array to the pool. The heap must not be used afterwards.</summary>
    public void Dispose()
    {
        if (_items.Length != 0)
        {
            ArrayPool<(float, int)>.Shared.Return(_items);
        }

        _items = [];
        _count = 0;
    }

    private void Grow()
    {
        int newCapacity = (int)Math.Min((long)_items.Length * 2, Array.MaxLength);
        if (newCapacity == _items.Length)
        {
            throw new OutOfMemoryException("DistanceHeap cannot grow further.");
        }

        var grown = ArrayPool<(float, int)>.Shared.Rent(newCapacity);
        Array.Copy(_items, grown, _count);
        ArrayPool<(float, int)>.Shared.Return(_items);
        _items = grown;
    }

    private void SiftDown(int start, (float Priority, int Id) item, int count)
    {
        int i = start;
        for (;;)
        {
            int left = 2 * i + 1;
            if (left >= count)
            {
                break;
            }

            int right = left + 1;
            int best = right < count && IsBetter(_items[right], _items[left]) ? right : left;
            if (!IsBetter(_items[best], item))
            {
                break;
            }

            _items[i] = _items[best];
            i = best;
        }

        _items[i] = item;
    }

    private readonly bool IsBetter((float Priority, int Id) candidate, (float Priority, int Id) incumbent) => _order switch
    {
        HeapOrder.MinIdAscending => candidate.Priority < incumbent.Priority
            || (candidate.Priority == incumbent.Priority && candidate.Id < incumbent.Id),
        HeapOrder.MaxIdAscending => candidate.Priority > incumbent.Priority
            || (candidate.Priority == incumbent.Priority && candidate.Id < incumbent.Id),
        HeapOrder.MaxIdDescending => candidate.Priority > incumbent.Priority
            || (candidate.Priority == incumbent.Priority && candidate.Id > incumbent.Id),
        HeapOrder.MinIdDescending => candidate.Priority < incumbent.Priority
            || (candidate.Priority == incumbent.Priority && candidate.Id > incumbent.Id),
        _ => throw new InvalidOperationException($"Unknown heap order '{_order}'."),
    };
}
