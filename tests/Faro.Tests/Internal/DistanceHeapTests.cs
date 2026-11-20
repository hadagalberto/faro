using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Internal;

public class DistanceHeapTests
{
    [Fact]
    public void Rent_InvalidCapacity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DistanceHeap.Rent(HeapOrder.Min, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => DistanceHeap.Rent(HeapOrder.Max, -1));
    }

    [Fact]
    public void MinHeap_PopsAscendingPriorities()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.Min, 8);

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
        using var heap = DistanceHeap.Rent(HeapOrder.Max, 8);

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
        using var heap = DistanceHeap.Rent(HeapOrder.Min, 8);
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
        using var heap = DistanceHeap.Rent(HeapOrder.Max, 8);
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

    [Fact]
    public void PushBounded_MaxHeap_KeepsSmallestPriorities_EvictingWorstFirst()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.Max, 8);

        // limit 3: the heap keeps the three smallest priorities seen so far; the root is the
        // worst kept and is the one replaced when something better arrives.
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

        // Remaining: (5,1), (6,5), (3,3); max-heap pops worst first: 6, 5, 3.
        Assert.True(heap.TryPop(out float p1, out int id1));
        Assert.Equal((6f, 5), (p1, id1));
        Assert.True(heap.TryPop(out float p2, out int id2));
        Assert.Equal((5f, 1), (p2, id2));
        Assert.True(heap.TryPop(out float p3, out int id3));
        Assert.Equal((3f, 3), (p3, id3));
    }

    [Fact]
    public void PushBounded_MinHeap_KeepsLargestPriorities_EvictingWorstFirst()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.Min, 8);

        Assert.True(heap.PushBounded(2, 1f, 0));
        Assert.True(heap.PushBounded(2, 4f, 1));
        Assert.False(heap.PushBounded(2, 3f, 2));  // worse (smaller) than root 4 -> dropped
        Assert.True(heap.PushBounded(2, 9f, 3));    // beats root 1 -> replaces it
        Assert.False(heap.PushBounded(2, 8f, 4));  // worse than root 8 -> dropped
        Assert.True(heap.PushBounded(2, 8.5f, 5));  // beats root 4 -> replaces it
        Assert.Equal(2, heap.Count);

        // Remaining: (9,3), (8.5,5); min-heap pops worst (smallest) first.
        Assert.True(heap.TryPop(out float p1, out int id1));
        Assert.Equal((8.5f, 5), (p1, id1));
        Assert.True(heap.TryPop(out float p2, out int id2));
        Assert.Equal((9f, 3), (p2, id2));
    }

    [Fact]
    public void PushBounded_Ties_KeepLowerId()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.Max, 4);
        Assert.True(heap.PushBounded(2, 5f, 10));
        Assert.True(heap.PushBounded(2, 5f, 11));
        // Equal priority, higher id loses against the root (which holds the lower id among ties).
        Assert.False(heap.PushBounded(2, 5f, 12));
        // Equal priority, lower id beats the root and replaces it.
        Assert.True(heap.PushBounded(2, 5f, 9));

        Assert.True(heap.TryPop(out float p1, out int id1));
        Assert.Equal((5f, 9), (p1, id1));
        Assert.True(heap.TryPop(out float p2, out int id2));
        Assert.Equal((5f, 11), (p2, id2));
    }

    [Fact]
    public void PushBounded_InvalidLimit_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            using var heap = DistanceHeap.Rent(HeapOrder.Min, 4);
            heap.PushBounded(0, 1f, 0);
        });
    }

    [Fact]
    public void Push_GrowsBeyondRentedCapacity()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.Min, 2);
        Assert.Equal(2, heap.Capacity);

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
        using var heap = DistanceHeap.Rent(HeapOrder.Max, 2);
        for (int i = 0; i < 100; i++)
        {
            Assert.True(heap.PushBounded(50, i, i));
        }

        // Keeps the fifty largest priorities: 50..99, popped worst-first: 99 down to 50.
        for (int i = 99; i >= 50; i--)
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
        using var heap = DistanceHeap.Rent(HeapOrder.Min, 8);
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

        for (int i = 10; i < 20; i++)
        {
            Assert.True(heap.TryPop(out float priority, out int id));
            Assert.Equal(i, id);
            FloatAssertions.Equal(-i, priority, 0f);
        }
    }

    [Fact]
    public void TryPeekRoot_DoesNotRemove()
    {
        using var heap = DistanceHeap.Rent(HeapOrder.Min, 4);
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
        using var minHeap = DistanceHeap.Rent(HeapOrder.Min, 4);
        Assert.False(minHeap.IsBetterThanRoot(0f, 0));
        minHeap.Push(5f, 7);
        Assert.True(minHeap.IsBetterThanRoot(4.9f, 1));
        Assert.False(minHeap.IsBetterThanRoot(5.1f, 1));
        Assert.False(minHeap.IsBetterThanRoot(5f, 8)); // tie: higher id loses
        Assert.True(minHeap.IsBetterThanRoot(5f, 3));  // tie: lower id wins

        using var maxHeap = DistanceHeap.Rent(HeapOrder.Max, 4);
        Assert.False(maxHeap.IsBetterThanRoot(0f, 0));
        maxHeap.Push(5f, 7);
        Assert.True(maxHeap.IsBetterThanRoot(5.1f, 1));
        Assert.False(maxHeap.IsBetterThanRoot(4.9f, 1));
        Assert.False(maxHeap.IsBetterThanRoot(5f, 8));
        Assert.True(maxHeap.IsBetterThanRoot(5f, 3));
    }

    [Fact]
    public void RentedHeaps_AreReusable_AcrossDisposeRentCycles()
    {
        for (int cycle = 0; cycle < 3; cycle++)
        {
            using var heap = DistanceHeap.Rent(HeapOrder.Min, 64);
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

    [Fact]
    public void RandomizedOperations_MatchSortedListModel()
    {
        const int operations = 600;

        for (int trial = 0; trial < 2; trial++)
        {
            HeapOrder order = trial == 0 ? HeapOrder.Min : HeapOrder.Max;
            using var heap = DistanceHeap.Rent(order, 16);
            var model = new List<(float Priority, int Id)>();
            ulong state = DeterministicRandom.DeriveSeed(0x5eed_6003, trial);

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
                    var expected = order == HeapOrder.Min
                        ? model.OrderBy(x => x.Priority).ThenBy(x => x.Id).First()
                        : model.OrderByDescending(x => x.Priority).ThenBy(x => x.Id).First();

                    Assert.True(heap.TryPop(out float priority, out int id));
                    FloatAssertions.Equal(expected.Priority, priority, 0f);
                    Assert.Equal(expected.Id, id);
                    model.Remove(expected);
                }
            }

            Assert.Equal(model.Count, heap.Count);
        }
    }

    [Fact]
    public void RandomizedPushBounded_MatchesTopKModel()
    {
        const int limit = 7;

        for (int trial = 0; trial < 2; trial++)
        {
            HeapOrder order = trial == 0 ? HeapOrder.Min : HeapOrder.Max;
            using var heap = DistanceHeap.Rent(order, limit);
            var model = new List<(float Priority, int Id)>();
            ulong state = DeterministicRandom.DeriveSeed(0x5eed_6004, trial);

            for (int id = 0; id < 800; id++)
            {
                float priority = DeterministicRandom.NextFloat(ref state) * 100f;
                bool accepted = heap.PushBounded(limit, priority, id);

                model.Add((priority, id));
                var best = order == HeapOrder.Min
                    ? model.OrderBy(x => x.Priority).ThenBy(x => x.Id).Take(limit).ToList()
                    : model.OrderByDescending(x => x.Priority).ThenBy(x => x.Id).Take(limit).ToList();

                Assert.Equal(best.Count, heap.Count);
                Assert.Equal(best.Contains((priority, id)), accepted);
            }

            // Final contents must be exactly the model's top-k, drained worst-first.
            var expected = order == HeapOrder.Min
                ? model.OrderBy(x => x.Priority).ThenBy(x => x.Id).Take(limit).Reverse().ToList()
                : model.OrderByDescending(x => x.Priority).ThenBy(x => x.Id).Take(limit).Reverse().ToList();

            foreach (var item in expected)
            {
                Assert.True(heap.TryPop(out float priority, out int id));
                FloatAssertions.Equal(item.Priority, priority, 0f);
                Assert.Equal(item.Id, id);
            }

            Assert.False(heap.TryPop(out _, out _));
        }
    }
}
