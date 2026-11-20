using Faro.Filtering;
using Xunit;

namespace Faro.Tests.Filtering;

public class SearchPlannerTests
{
    [Fact]
    public void OnePercentSelectivity_UsesExactCandidates()
    {
        var planner = new SearchPlanner();

        SearchDecision decision = planner.Plan(candidateCount: 100, totalCount: 10_000, baseEf: 64);

        Assert.Equal(SearchStrategy.ExactCandidates, decision.Strategy);
        Assert.Equal(0.01, decision.Selectivity, 6);
        Assert.Equal(64, decision.EffectiveEf);
    }

    [Fact]
    public void BroadFilter_UsesTraversalWithAdaptiveEf()
    {
        var planner = new SearchPlanner();

        SearchDecision decision = planner.Plan(candidateCount: 5_000, totalCount: 10_000, baseEf: 64);

        Assert.Equal(SearchStrategy.FilteredTraversal, decision.Strategy);
        Assert.Equal(0.5, decision.Selectivity, 6);
        Assert.Equal(128, decision.EffectiveEf);
    }

    [Fact]
    public void AdaptiveEf_IsCappedByMaxEf()
    {
        var planner = new SearchPlanner(exactThreshold: 0.005, maxEf: 256);

        SearchDecision decision = planner.Plan(candidateCount: 100, totalCount: 10_000, baseEf: 64);

        Assert.Equal(SearchStrategy.FilteredTraversal, decision.Strategy);
        Assert.Equal(256, decision.EffectiveEf);
    }

    [Fact]
    public void EmptyCollectionOrNoCandidates_UsesExact()
    {
        var planner = new SearchPlanner();

        Assert.Equal(SearchStrategy.ExactCandidates, planner.Plan(0, 0, 64).Strategy);
        Assert.Equal(SearchStrategy.ExactCandidates, planner.Plan(0, 500, 64).Strategy);
        Assert.Equal(0, planner.Plan(0, 500, 64).Selectivity);
    }

    [Fact]
    public void ThresholdBoundary_IsInclusive()
    {
        var planner = new SearchPlanner(exactThreshold: 0.1, maxEf: 4096);

        Assert.Equal(SearchStrategy.ExactCandidates, planner.Plan(10, 100, 32).Strategy);
        Assert.Equal(SearchStrategy.FilteredTraversal, planner.Plan(11, 100, 32).Strategy);
    }

    [Fact]
    public void InvalidArguments_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SearchPlanner(0, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SearchPlanner(1.1, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SearchPlanner(0.1, 0));

        var planner = new SearchPlanner();
        Assert.Throws<ArgumentOutOfRangeException>(() => planner.Plan(-1, 10, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() => planner.Plan(1, -1, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() => planner.Plan(1, 10, 0));
        Assert.Throws<ArgumentException>(() => planner.Plan(11, 10, 64));
    }
}
