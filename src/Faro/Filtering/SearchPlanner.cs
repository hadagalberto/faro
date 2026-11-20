namespace Faro.Filtering;

/// <summary>Execution strategy chosen for a filtered vector search.</summary>
public enum SearchStrategy
{
    /// <summary>Score exactly the ids returned by the payload indexes (best when few vectors match).</summary>
    ExactCandidates,

    /// <summary>Traverse the HNSW graph applying the filter during search, with an expanded beam.</summary>
    FilteredTraversal,
}

/// <summary>
/// The planner's decision for one filtered search.
/// </summary>
/// <param name="Strategy">The chosen execution strategy.</param>
/// <param name="CandidateCount">Number of ids matching the filter.</param>
/// <param name="TotalCount">Number of live vectors in the collection.</param>
/// <param name="Selectivity">Fraction of live vectors matching the filter, in [0, 1].</param>
/// <param name="EffectiveEf">Beam width to use for filtered traversal.</param>
public readonly record struct SearchDecision(
    SearchStrategy Strategy,
    int CandidateCount,
    int TotalCount,
    double Selectivity,
    int EffectiveEf);

/// <summary>
/// Chooses between exact scoring over the filtered candidate set and filtered HNSW traversal.
/// </summary>
/// <remarks>
/// Selectivity is the fraction of live vectors matching the filter. When it is at or below
/// <see cref="ExactThreshold"/> the candidate list is smaller than a graph walk and is scored
/// exactly, giving perfect filtered recall. Above the threshold the graph is traversed with the
/// filter applied at visit time; the beam is widened by <c>1 / selectivity</c> (capped at
/// <see cref="MaxEf"/>) so sparse matching regions are still explored thoroughly.
/// </remarks>
public sealed class SearchPlanner
{
    /// <summary>Creates a planner with the default 5% exact threshold and 4096 beam cap.</summary>
    public SearchPlanner()
        : this(0.05, 4096)
    {
    }

    /// <summary>Creates a planner with explicit tuning.</summary>
    /// <param name="exactThreshold">Selectivity at or below which exact candidate scoring is used; must be in (0, 1].</param>
    /// <param name="maxEf">Upper bound for the adaptive beam; must be positive.</param>
    /// <exception cref="ArgumentOutOfRangeException">A parameter is outside its valid range.</exception>
    public SearchPlanner(double exactThreshold, int maxEf)
    {
        if (double.IsNaN(exactThreshold) || exactThreshold <= 0 || exactThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(exactThreshold), exactThreshold, "threshold must be in (0, 1].");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxEf);

        ExactThreshold = exactThreshold;
        MaxEf = maxEf;
    }

    /// <summary>Selectivity at or below which exact candidate scoring is used.</summary>
    public double ExactThreshold { get; }

    /// <summary>Upper bound for the adaptive beam during filtered traversal.</summary>
    public int MaxEf { get; }

    /// <summary>Plans one filtered search.</summary>
    /// <param name="candidateCount">Number of ids matching the filter.</param>
    /// <param name="totalCount">Total number of live vectors.</param>
    /// <param name="baseEf">The search beam configured for the collection.</param>
    /// <returns>The decision, including the adaptive beam width.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A count is negative or <paramref name="baseEf"/> is not positive.</exception>
    public SearchDecision Plan(int candidateCount, int totalCount, int baseEf)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(candidateCount);
        ArgumentOutOfRangeException.ThrowIfNegative(totalCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(baseEf);

        if (candidateCount > totalCount)
        {
            throw new ArgumentException("candidateCount cannot exceed totalCount", nameof(candidateCount));
        }

        double selectivity = totalCount == 0 ? 0 : (double)candidateCount / totalCount;
        if (candidateCount == 0 || selectivity <= ExactThreshold)
        {
            return new SearchDecision(SearchStrategy.ExactCandidates, candidateCount, totalCount, selectivity, baseEf);
        }

        double widened = Math.Ceiling(baseEf / Math.Max(selectivity, 1e-6));
        int effectiveEf = (int)Math.Clamp(widened, baseEf, MaxEf);
        return new SearchDecision(SearchStrategy.FilteredTraversal, candidateCount, totalCount, selectivity, effectiveEf);
    }
}
