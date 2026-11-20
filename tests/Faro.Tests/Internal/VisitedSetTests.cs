using Faro.Internal;
using Faro.Testing;
using Xunit;

namespace Faro.Tests.Internal;

public class VisitedSetTests
{
    [Fact]
    public void Rent_InvalidCapacity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => VisitedSet.Rent(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => VisitedSet.Rent(-1));
    }

    [Fact]
    public void FreshSet_MarksNothing()
    {
        using var set = VisitedSet.Rent(32);
        Assert.Equal(0, set.Count);
        for (int id = 0; id < 32; id++)
        {
            Assert.False(set.Contains(id));
        }
    }

    [Fact]
    public void Add_MarksIdOnce()
    {
        using var set = VisitedSet.Rent(32);
        Assert.True(set.Add(7));
        Assert.True(set.Contains(7));
        Assert.False(set.Add(7));
        Assert.True(set.Contains(7));
        Assert.Equal(1, set.Count);
    }

    [Fact]
    public void Add_NegativeId_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            using var set = VisitedSet.Rent(8);
            set.Add(-1);
        });
    }

    [Fact]
    public void Contains_NegativeId_IsFalse()
    {
        using var set = VisitedSet.Rent(8);
        Assert.False(set.Contains(-1));
        Assert.False(set.Contains(int.MinValue));
    }

    [Fact]
    public void Reset_ClearsAllMarksInConstantTime()
    {
        using var set = VisitedSet.Rent(64);
        for (int id = 0; id < 64; id++)
        {
            set.Add(id);
        }

        Assert.Equal(64, set.Count);
        set.Reset();
        Assert.Equal(0, set.Count);

        for (int id = 0; id < 64; id++)
        {
            Assert.False(set.Contains(id));
        }

        // The set is fully usable again after a reset.
        Assert.True(set.Add(13));
        Assert.True(set.Contains(13));
        Assert.Equal(1, set.Count);
    }

    [Fact]
    public void Add_GrowsBeyondInitialCapacity()
    {
        using var set = VisitedSet.Rent(4);
        for (int id = 0; id < 500; id++)
        {
            Assert.True(set.Add(id));
        }

        Assert.Equal(500, set.Count);
        for (int id = 0; id < 500; id++)
        {
            Assert.True(set.Contains(id));
            Assert.False(set.Add(id));
        }
    }

    [Fact]
    public void Reset_RepeatedManyTimes_StaysConsistent()
    {
        using var set = VisitedSet.Rent(16);
        for (int cycle = 0; cycle < 1000; cycle++)
        {
            set.Reset();
            Assert.Equal(0, set.Count);
            Assert.True(set.Add(cycle % 16));
            Assert.True(set.Contains(cycle % 16));
            Assert.Equal(1, set.Count);
        }
    }

    [Fact]
    public void VersionOverflow_HardClearsAndRestarts()
    {
        // Direct construction at the generation ceiling exercises the overflow branch of Reset
        // without performing two billion resets.
        var set = new VisitedSet(new int[8], int.MaxValue);
        Assert.True(set.Add(3));
        Assert.True(set.Contains(3));
        Assert.Equal(1, set.Count);

        set.Reset();
        Assert.Equal(0, set.Count);
        Assert.False(set.Contains(3));

        // The stamp counter restarted at 1, so the set keeps working.
        Assert.True(set.Add(3));
        Assert.True(set.Add(5));
        Assert.Equal(2, set.Count);
        Assert.True(set.Contains(3));
        Assert.True(set.Contains(5));
        set.Dispose();
    }

    [Fact]
    public void RentedSets_AreReusable_AcrossDisposeRentCycles()
    {
        for (int cycle = 0; cycle < 3; cycle++)
        {
            using var set = VisitedSet.Rent(64);
            Assert.Equal(0, set.Count);
            Assert.True(set.Add(cycle * 3));
            Assert.True(set.Contains(cycle * 3));
            Assert.False(set.Contains(cycle * 3 + 1));
        }
    }

    [Fact]
    public void RandomizedAdds_MatchHashSetModel()
    {
        using var set = VisitedSet.Rent(16);
        var model = new HashSet<int>();
        ulong state = 0x5eed_7001;

        for (int op = 0; op < 2000; op++)
        {
            if (op % 25 == 24)
            {
                set.Reset();
                model.Clear();
                continue;
            }

            int id = DeterministicRandom.NextInt(ref state, 2000);
            Assert.Equal(model.Add(id), set.Add(id));
            Assert.Equal(model.Contains(id), set.Contains(id));
        }

        Assert.Equal(model.Count, set.Count);
    }
}
