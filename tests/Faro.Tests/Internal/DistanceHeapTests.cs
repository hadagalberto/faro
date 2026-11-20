using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Internal;

/// <summary>
/// Contract tests for the pooled top-k heap. Every order is a total order on (priority, id);
/// the heap root is its maximum. <see cref="DistanceHeap.PushBounded"/> therefore keeps the
/// <c>limit</c> items that are <em>worst</em> under the order (the inverse order's best-k),
/// which is exactly what index searches need: they rent the inverse of their metric order so
/// the root is the worst kept hit and evictions are metric-correct even at score ties.
/// </summary>
public class DistanceHeapTests
{
    // HeapOrder is internal to Faro, so theories pass its integer value and cast inside.
    public static TheoryData<int> Orders { get; } = new()
    {
        (int)HeapOrder.MinIdAscending,
        (int)HeapOrder.MaxIdAscending,
        (int)HeapOrder.MaxIdDescending,
        (int)HeapOrder.MinIdDescending,
    };

    [Fact]
    public void Rent_InvalidCapacity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DistanceHeap.Rent(HeapOrder.MinIdAscending, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => DistanceHeap.Rent(HeapOrder.MaxIdAscending, -1));
    }

    [Fact]
    public void MinHeap_PopsAscendingPriorities()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MinIdAscending, 8);

        ulong state = 0x5eed_6001;
        var pushed = new List<(float Priority, int Id)>();
        for (int i = 0; i < 200; i++)
        {
            float priority = DeterministicRandom.NextFloat(ref state) * 100f;
            pushed.Add((priority, i));
            heap.Push(priority, i);
        }

        var expected = pushed.OrderBy(x => x.Priority).ThenBy(x => x.Id).ToList();
        foreach (var item in expected)
        {
            Assert.True(heap.TryPop(out float priority, out int id));
            FloatAssertions.Equal(item.Priority, priority, 0f);
            Assert.Equal(item.Id, id);
        }

        Assert.False(heap.TryPop(out _, out _));
        Assert.Equal(0, heap.Count);
    }

    [Fact]
    public void MaxHeap_PopsDescendingPriorities()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MaxIdAscending, 8);

        ulong state = 0x5eed_6002;
        var pushed = new List<(float Priority, int Id)>();
        for (int i = 0; i < 200; i++)
        {
            float priority = DeterministicRandom.NextFloat(ref state) * 100f;
            pushed.Add((priority, i));
            heap.Push(priority, i);
        }

        var expected = pushed.OrderByDescending(x => x.Priority).ThenBy(x => x.Id).ToList();
        foreach (var item in expected)
        {
            Assert.True(heap.TryPop(out float priority, out int id));
            FloatAssertions.Equal(item.Priority, priority, 0f);
            Assert.Equal(item.Id, id);
        }

        Assert.False(heap.TryPop(out _, out _));
        Assert.Equal(0, heap.Count);
    }

    [Fact]
    public void MinHeap_Ties_BreakByAscendingId()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MinIdAscending, 8);
        heap.Push(5f, 7);
        heap.Push(5f, 2);
        heap.Push(5f, 9);
        heap.Push(5f, 1);
        Assert.Equal(4, heap.Count);

        Assert.True(heap.TryPop(out float p1, out int id1));
        Assert.Equal((5f, 1), (p1, id1));
        Assert.True(heap.TryPop(out float p2, out int id2));
        Assert.Equal((5f, 2), (p2, id2));
        Assert.True(heap.TryPop(out float p3, out int id3));
        Assert.Equal((5f, 7), (p3, id3));
        Assert.True(heap.TryPop(out float p4, out int id4));
        Assert.Equal((5f, 9), (p4, id4));
    }

    [Fact]
    public void MaxHeap_Ties_BreakByAscendingId()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MaxIdAscending, 8);
        heap.Push(5f, 7);
        heap.Push(5f, 2);
        heap.Push(5f, 9);
        heap.Push(5f, 1);

        Assert.True(heap.TryPop(out float p1, out int id1));
        Assert.Equal((5f, 1), (p1, id1));
        Assert.True(heap.TryPop(out float p2, out int id2));
        Assert.Equal((5f, 2), (p2, id2));
        Assert.True(heap.TryPop(out float p3, out int id3));
        Assert.Equal((5f, 7), (p3, id3));
        Assert.True(heap.TryPop(out float p4, out int id4));
        Assert.Equal((5f, 9), (p4, id4));
    }

    [Theory]
    [InlineData((int)HeapOrder.MaxIdDescending)]
    [InlineData((int)HeapOrder.MinIdDescending)]
    public void IdDescendingOrders_Ties_BreakByDescendingId(int orderValue)
    {
        var order = (HeapOrder)orderValue;
        using var heap = DistanceHeap.Rent(order, 8);
        foreach (int id in new[] { 7, 2, 9, 1 })
        {
            heap.Push(5f, id);
        }

        foreach (int expectedId in new[] { 9, 7, 2, 1 })
        {
            Assert.True(heap.TryPop(out float priority, out int id));
            Assert.Equal((5f, expectedId), (priority, id));
        }
    }

    [Fact]
    public void PushBounded_MaxIdDescending_KeepsSmallestPriorities_EvictingWorstFirst()
    {
        // This is the order FlatIndex rents for lower-is-closer metrics: the root is the
        // largest priority (the worst kept hit), so a closer candidate replaces it.
        using var heap = DistanceHeap.Rent(HeapOrder.MaxIdDescending, 8);

        Assert.True(heap.PushBounded(3, 10f, 0));
        Assert.True(heap.PushBounded(3, 5f, 1));
        Assert.True(heap.PushBounded(3, 8f, 2));
        Assert.Equal(3, heap.Count);

        Assert.True(heap.PushBounded(3, 3f, 3));    // beats 10 -> replaces it
        Assert.True(heap.PushBounded(3, 7f, 4));    // beats 8  -> replaces it
        Assert.True(heap.PushBounded(3, 6f, 5));    // beats 7  -> replaces it
        Assert.False(heap.PushBounded(3, 9f, 6));   // worse than root 6 -> dropped
        Assert.False(heap.PushBounded(3, 6.5f, 7)); // worse than root 6 -> dropped
        Assert.Equal(3, heap.Count);

        // Remaining: (5,1), (6,5), (3,3); popped worst-first: 6, 5, 3.
        Assert.True(heap.TryPop(out float p1, out int id1));
        Assert.Equal((6f, 5), (p1, id1));
        Assert.True(heap.TryPop(out float p2, out int id2));
        Assert.Equal((5f, 1), (p2, id2));
        Assert.True(heap.TryPop(out float p3, out int id3));
        Assert.Equal((3f, 3), (p3, id3));
    }

    [Fact]
    public void PushBounded_MinIdDescending_KeepsLargestPriorities_EvictingWorstFirst()
    {
        // The inverse order FlatIndex rents for DotProduct: the root is the smallest
        // similarity (the worst kept hit), so a more similar candidate replaces it.
        using var heap = DistanceHeap.Rent(HeapOrder.MinIdDescending, 8);

        Assert.True(heap.PushBounded(2, 1f, 0));
        Assert.True(heap.PushBounded(2, 4f, 1));
        Assert.True(heap.PushBounded(2, 3f, 2));    // beats 1 -> replaces it
        Assert.True(heap.PushBounded(2, 9f, 3));    // beats 3 -> replaces it
        Assert.True(heap.PushBounded(2, 8f, 4));    // beats 4 -> replaces it
        Assert.True(heap.PushBounded(2, 8.5f, 5));  // beats 8 -> replaces it
        Assert.Equal(2, heap.Count);

        // Remaining: (9,3), (8.5,5); popped worst-first: 8.5, then 9.
        Assert.True(heap.TryPop(out float p1, out int id1));
        Assert.Equal((8.5f, 5), (p1, id1));
        Assert.True(heap.TryPop(out float p2, out int id2));
        Assert.Equal((9f, 3), (p2, id2));
    }

    [Fact]
    public void PushBounded_MaxIdDescending_Ties_KeepLowerId()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MaxIdDescending, 4);
        Assert.True(heap.PushBounded(2, 5f, 10));
        Assert.True(heap.PushBounded(2, 5f, 11));
        // Root holds the higher id among ties, so a higher id is dropped and a lower one wins.
        Assert.False(heap.PushBounded(2, 5f, 12));
        Assert.True(heap.PushBounded(2, 5f, 9));

        Assert.Equal(2, heap.Count);
        Assert.True(heap.TryPop(out float p1, out int id1));
        Assert.Equal((5f, 10), (p1, id1));
        Assert.True(heap.TryPop(out float p2, out int id2));
        Assert.Equal((5f, 9), (p2, id2));
    }

    [Fact]
    public void PushBounded_BelowLimit_AlwaysAccepts()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MaxIdAscending, 4);
        for (int i = 0; i < 4; i++)
        {
            Assert.True(heap.PushBounded(4, -i, i));
        }

        Assert.Equal(4, heap.Count);
    }

    [Fact]
    public void PushBounded_InvalidLimit_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            using var heap = DistanceHeap.Rent(HeapOrder.MinIdAscending, 4);
            heap.PushBounded(0, 1f, 0);
        });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            using var heap = DistanceHeap.Rent(HeapOrder.MaxIdDescending, 4);
            heap.PushBounded(-1, 1f, 0);
        });
    }

    [Fact]
    public void Push_GrowsBeyondRentedCapacity()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MinIdAscending, 2);
        // ArrayPool rents a whole size bucket, so capacity is only guaranteed to be >= request.
        Assert.True(heap.Capacity >= 2);

        for (int i = 0; i < 500; i++)
        {
            heap.Push(i, i);
            Assert.Equal(i + 1, heap.Count);
        }

        Assert.True(heap.Capacity >= 500);
        for (int i = 0; i < 500; i++)
        {
            Assert.True(heap.TryPop(out float priority, out int id));
            Assert.Equal(i, id);
            FloatAssertions.Equal(i, priority, 0f);
        }
    }

    [Fact]
    public void PushBounded_GrowsLimitBeyondRentedCapacity()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MinIdDescending, 2);
        for (int i = 0; i < 100; i++)
        {
            Assert.True(heap.PushBounded(50, i, i));
        }

        // Keeps the fifty largest priorities: 50..99. The root is the worst kept (smallest),
        // so popping yields them in ascending order.
        for (int i = 50; i < 100; i++)
        {
            Assert.True(heap.TryPop(out float priority, out int id));
            Assert.Equal(i, id);
            FloatAssertions.Equal(i, priority, 0f);
        }

        Assert.False(heap.TryPop(out _, out _));
    }

    [Fact]
    public void Clear_AllowsFullReuse()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MinIdAscending, 8);
        for (int i = 0; i < 10; i++)
        {
            heap.Push(i, i);
        }

        heap.Clear();
        Assert.Equal(0, heap.Count);
        Assert.False(heap.TryPop(out _, out _));

        for (int i = 10; i < 20; i++)
        {
            heap.Push(-i, i);
        }

        // Min-heap: the most negative priority (-19) is the root and pops first.
        for (int i = 19; i >= 10; i--)
        {
            Assert.True(heap.TryPop(out float priority, out int id));
            Assert.Equal(i, id);
            FloatAssertions.Equal(-i, priority, 0f);
        }
    }

    [Fact]
    public void TryPeekRoot_DoesNotRemove()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.MinIdAscending, 4);
        Assert.False(heap.TryPeekRoot(out _, out _));

        heap.Push(7f, 3);
        heap.Push(2f, 8);

        Assert.True(heap.TryPeekRoot(out float priority, out int id));
        Assert.Equal((2f, 8), (priority, id));
        Assert.Equal(2, heap.Count);
    }

    [Fact]
    public void IsBetterThanRoot_FollowsHeapOrder()
    {
        using var minAscending = DistanceHeap.Rent(HeapOrder.MinIdAscending, 4);
        Assert.False(minAscending.IsBetterThanRoot(0f, 0)); // empty heap has no root
        minAscending.Push(5f, 7);
        Assert.True(minAscending.IsBetterThanRoot(4.9f, 1));
        Assert.False(minAscending.IsBetterThanRoot(5.1f, 1));
        Assert.False(minAscending.IsBetterThanRoot(5f, 8)); // tie: higher id loses
        Assert.True(minAscending.IsBetterThanRoot(5f, 3));  // tie: lower id wins

        using var maxAscending = DistanceHeap.Rent(HeapOrder.MaxIdAscending, 4);
        Assert.False(maxAscending.IsBetterThanRoot(0f, 0));
        maxAscending.Push(5f, 7);
        Assert.True(maxAscending.IsBetterThanRoot(5.1f, 1));
        Assert.False(maxAscending.IsBetterThanRoot(4.9f, 1));
        Assert.False(maxAscending.IsBetterThanRoot(5f, 8));
        Assert.True(maxAscending.IsBetterThanRoot(5f, 3));

        using var maxDescending = DistanceHeap.Rent(HeapOrder.MaxIdDescending, 4);
        maxDescending.Push(5f, 7);
        Assert.True(maxDescending.IsBetterThanRoot(5f, 8)); // tie: higher id wins
        Assert.False(maxDescending.IsBetterThanRoot(5f, 3));
        Assert.True(maxDescending.IsBetterThanRoot(5.1f, 3));

        using var minDescending = DistanceHeap.Rent(HeapOrder.MinIdDescending, 4);
        minDescending.Push(5f, 7);
        Assert.True(minDescending.IsBetterThanRoot(5f, 8));
        Assert.False(minDescending.IsBetterThanRoot(5f, 3));
        Assert.True(minDescending.IsBetterThanRoot(4.9f, 3));
    }

    [Fact]
    public void RentedHeaps_AreReusable_AcrossDisposeRentCycles()
    {
        for (int cycle = 0; cycle < 3; cycle++)
        {
            using var heap = DistanceHeap.Rent(HeapOrder.MinIdAscending, 64);
            Assert.Equal(0, heap.Count);
            for (int i = 99; i >= 0; i--)
            {
                heap.Push(i, i);
            }

            for (int i = 0; i < 100; i++)
            {
                Assert.True(heap.TryPop(out float priority, out int id));
                Assert.Equal(i, id);
            }
        }
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public void RandomizedOperations_MatchSortedListModel(int orderValue)
    {
        var order = (HeapOrder)orderValue;
        const int operations = 600;

        using var heap = DistanceHeap.Rent(order, 16);
        var model = new List<(float Priority, int Id)>();
        ulong state = DeterministicRandom.DeriveSeed(0x5eed_6003, (int)order);

        for (int op = 0; op < operations; op++)
        {
            int kind = DeterministicRandom.NextInt(ref state, 3);
            if (kind != 0 || model.Count == 0)
            {
                float priority = DeterministicRandom.NextFloat(ref state) * 100f;
                heap.Push(priority, op);
                model.Add((priority, op));
            }
            else
            {
                model.Sort((a, b) => CompareBestFirst(order, a, b));
                var expected = model[0];

                Assert.True(heap.TryPop(out float priority, out int id));
                FloatAssertions.Equal(expected.Priority, priority, 0f);
                Assert.Equal(expected.Id, id);
                model.RemoveAt(0);
            }
        }

        Assert.Equal(model.Count, heap.Count);
        model.Sort((a, b) => CompareBestFirst(order, a, b));
        foreach (var expected in model)
        {
            Assert.True(heap.TryPop(out float priority, out int id));
            FloatAssertions.Equal(expected.Priority, priority, 0f);
            Assert.Equal(expected.Id, id);
        }

        Assert.False(heap.TryPop(out _, out _));
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public void RandomizedPushBounded_MatchesTopKModel(int orderValue)
    {
        var order = (HeapOrder)orderValue;
        const int limit = 7;

        using var heap = DistanceHeap.Rent(order, limit);
        var model = new List<(float Priority, int Id)>();
        ulong state = DeterministicRandom.DeriveSeed(0x5eed_6004, (int)order);

        for (int id = 0; id < 800; id++)
        {
            float priority = DeterministicRandom.NextFloat(ref state) * 100f;
            bool accepted = heap.PushBounded(limit, priority, id);

            model.Add((priority, id));
            model.Sort((a, b) => CompareWorstFirst(order, a, b));
            var expected = model.Take(limit).ToList();

            Assert.Equal(expected.Count, heap.Count);
            Assert.Equal(expected.Contains((priority, id)), accepted);
        }

        // Draining yields the kept set best-first, i.e. the reverse of the worst-first model.
        var expectedPopOrder = model.Take(limit).ToList();
        expectedPopOrder.Sort((a, b) => CompareBestFirst(order, a, b));
        foreach (var item in expectedPopOrder)
        {
            Assert.True(heap.TryPop(out float priority, out int id));
            FloatAssertions.Equal(item.Priority, priority, 0f);
            Assert.Equal(item.Id, id);
        }

        Assert.False(heap.TryPop(out _, out _));
    }

    /// <summary>Orders pairs from worst to best under <paramref name="order"/>.</summary>
    private static int CompareWorstFirst(HeapOrder order, (float Priority, int Id) a, (float Priority, int Id) b)
    {
        int byPriority = order switch
        {
            HeapOrder.MinIdAscending or HeapOrder.MinIdDescending => b.Priority.CompareTo(a.Priority),
            HeapOrder.MaxIdAscending or HeapOrder.MaxIdDescending => a.Priority.CompareTo(b.Priority),
            _ => throw new InvalidOperationException($"Unknown heap order '{order}'."),
        };

        if (byPriority != 0)
        {
            return byPriority;
        }

        return order switch
        {
            HeapOrder.MinIdAscending or HeapOrder.MaxIdAscending => b.Id.CompareTo(a.Id),
            HeapOrder.MaxIdDescending or HeapOrder.MinIdDescending => a.Id.CompareTo(b.Id),
            _ => throw new InvalidOperationException($"Unknown heap order '{order}'."),
        };
    }

    private static int CompareBestFirst(HeapOrder order, (float Priority, int Id) a, (float Priority, int Id) b) =>
        -CompareWorstFirst(order, a, b);
}
