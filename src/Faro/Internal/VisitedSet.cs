using System.Buffers;

namespace Faro.Internal;

/// <summary>
/// A pooled, generation-stamped visited set used by graph searches to avoid re-visiting nodes
/// without per-search allocations.
/// </summary>
/// <remarks>
/// <para>
/// The backing stamp array is rented from <see cref="ArrayPool{T}"/> and cleared once at rent;
/// <see cref="Reset"/> just advances the generation stamp, which makes every previous mark
/// invisible in O(1). When the stamp counter wraps after <see cref="int.MaxValue"/> resets the
/// array is cleared once more and the generation restarts at 1.
/// </para>
/// </remarks>
internal ref struct VisitedSet
{
    private int[] _stamps;
    private int _version;
    private int _count;

    // Internal for the overflow-branch test, which places the generation at its ceiling.
    internal VisitedSet(int[] stamps, int version)
    {
        _stamps = stamps;
        _version = version;
        _count = 0;
    }

    /// <summary>Number of distinct ids currently marked.</summary>
    public int Count => _count;

    /// <summary>Rents a visited set able to hold ids in [0, capacity) without growing.</summary>
    public static VisitedSet Rent(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        int[] stamps = ArrayPool<int>.Shared.Rent(capacity);
        Array.Clear(stamps);
        return new VisitedSet(stamps, 1);
    }

    /// <summary>Marks an id as visited.</summary>
    /// <returns><see langword="true"/> if the id was not marked before.</returns>
    public bool Add(int id)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(id);
        EnsureCapacity(id + 1);
        if (_stamps[id] == _version)
        {
            return false;
        }

        _stamps[id] = _version;
        _count++;
        return true;
    }

    /// <summary>Returns whether an id is currently marked. Unknown (never added) ids are not marked.</summary>
    public bool Contains(int id) => id >= 0 && (uint)id < (uint)_stamps.Length && _stamps[id] == _version;

    /// <summary>Starts a new generation: all previous marks become invisible in O(1).</summary>
    public void Reset()
    {
        if (_version == int.MaxValue)
        {
            Array.Clear(_stamps);
            _version = 1;
        }
        else
        {
            _version++;
        }

        _count = 0;
    }

    /// <summary>Returns the backing array to the pool. The set must not be used afterwards.</summary>
    public void Dispose()
    {
        if (_stamps.Length != 0)
        {
            ArrayPool<int>.Shared.Return(_stamps);
        }

        _stamps = [];
        _version = 0;
        _count = 0;
    }

    private void EnsureCapacity(int needed)
    {
        if (needed <= _stamps.Length)
        {
            return;
        }

        int[] grown = ArrayPool<int>.Shared.Rent(Math.Max(needed, _stamps.Length * 2));
        Array.Clear(grown);
        Array.Copy(_stamps, grown, _stamps.Length);
        ArrayPool<int>.Shared.Return(_stamps);
        _stamps = grown;
    }
}
