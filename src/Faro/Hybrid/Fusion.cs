namespace Faro.Hybrid;

/// <summary>
/// Strategy used to merge a vector ranking and a BM25 ranking into one result list.
/// </summary>
public abstract class Fusion
{
    /// <summary>
    /// Creates reciprocal rank fusion (Cormack, Clarke &amp; Buettcher, 2009).
    /// </summary>
    /// <param name="k">Rank constant; larger values flatten the contribution of top ranks. Defaults to 60.</param>
    /// <returns>The fusion strategy.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="k"/> is not positive.</exception>
    public static Fusion Rrf(int k = 60)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        return new RrfFusion(k);
    }

    /// <summary>
    /// Creates normalized weighted score fusion. Each list is min-max normalized so its best hit
    /// scores 1, then combined with the supplied weights.
    /// </summary>
    /// <param name="vectorWeight">Weight of the normalized vector score.</param>
    /// <param name="textWeight">Weight of the normalized BM25 score.</param>
    /// <returns>The fusion strategy.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A weight is negative or their sum is zero.</exception>
    public static Fusion Weighted(double vectorWeight = 0.5, double textWeight = 0.5)
    {
        if (double.IsNaN(vectorWeight) || double.IsNaN(textWeight)
            || vectorWeight < 0 || textWeight < 0 || vectorWeight + textWeight == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(vectorWeight), "weights must be non-negative and not both zero");
        }

        return new WeightedFusion(vectorWeight, textWeight);
    }

    internal abstract HybridResult[] Fuse(
        IReadOnlyList<SearchResult> vectorHits,
        IReadOnlyList<(int DocId, double Score)> textHits,
        bool lowerIsBetterVectorScore,
        int limit);

    private sealed class RrfFusion(int k) : Fusion
    {
        internal override HybridResult[] Fuse(
            IReadOnlyList<SearchResult> vectorHits,
            IReadOnlyList<(int DocId, double Score)> textHits,
            bool lowerIsBetterVectorScore,
            int limit)
        {
            var merged = new Dictionary<int, Entry>();
            for (int i = 0; i < vectorHits.Count; i++)
            {
                SearchResult hit = vectorHits[i];
                Entry entry = GetOrAdd(merged, hit.Id);
                entry.Score += 1.0 / (k + i + 1);
                if (entry.VectorRank == 0)
                {
                    entry.VectorRank = i + 1;
                    entry.VectorScore = hit.Score;
                }
            }

            for (int i = 0; i < textHits.Count; i++)
            {
                (int docId, double score) = textHits[i];
                Entry entry = GetOrAdd(merged, docId);
                entry.Score += 1.0 / (k + i + 1);
                if (entry.TextRank == 0)
                {
                    entry.TextRank = i + 1;
                    entry.TextScore = score;
                }
            }

            return Finish(merged, limit);
        }

        private static Entry GetOrAdd(Dictionary<int, Entry> merged, int id)
        {
            if (!merged.TryGetValue(id, out Entry? entry))
            {
                entry = new Entry(id);
                merged.Add(id, entry);
            }

            return entry;
        }
    }

    private sealed class WeightedFusion(double vectorWeight, double textWeight) : Fusion
    {
        internal override HybridResult[] Fuse(
            IReadOnlyList<SearchResult> vectorHits,
            IReadOnlyList<(int DocId, double Score)> textHits,
            bool lowerIsBetterVectorScore,
            int limit)
        {
            var merged = new Dictionary<int, Entry>();

            if (vectorHits.Count != 0)
            {
                float min = vectorHits[0].Score;
                float max = min;
                for (int i = 1; i < vectorHits.Count; i++)
                {
                    min = Math.Min(min, vectorHits[i].Score);
                    max = Math.Max(max, vectorHits[i].Score);
                }

                for (int i = 0; i < vectorHits.Count; i++)
                {
                    SearchResult hit = vectorHits[i];
                    double normalized = Normalize(hit.Score, min, max, lowerIsBetterVectorScore);
                    Entry entry = GetOrAdd(merged, hit.Id);
                    entry.Score += vectorWeight * normalized;
                    entry.VectorRank = i + 1;
                    entry.VectorScore = hit.Score;
                }
            }

            if (textHits.Count != 0)
            {
                double min = textHits[0].Score;
                double max = min;
                for (int i = 1; i < textHits.Count; i++)
                {
                    min = Math.Min(min, textHits[i].Score);
                    max = Math.Max(max, textHits[i].Score);
                }

                for (int i = 0; i < textHits.Count; i++)
                {
                    (int docId, double score) = textHits[i];
                    double normalized = Normalize(score, min, max, lowerIsBetter: false);
                    Entry entry = GetOrAdd(merged, docId);
                    entry.Score += textWeight * normalized;
                    entry.TextRank = i + 1;
                    entry.TextScore = score;
                }
            }

            return Finish(merged, limit);
        }

        private static double Normalize(double value, double min, double max, bool lowerIsBetter) =>
            max == min ? 1.0 : lowerIsBetter ? (max - value) / (max - min) : (value - min) / (max - min);

        private static Entry GetOrAdd(Dictionary<int, Entry> merged, int id)
        {
            if (!merged.TryGetValue(id, out Entry? entry))
            {
                entry = new Entry(id);
                merged.Add(id, entry);
            }

            return entry;
        }
    }

    private sealed class Entry(int id)
    {
        internal int Id { get; } = id;

        internal double Score;

        internal float? VectorScore;

        internal double? TextScore;

        internal int VectorRank;

        internal int TextRank;
    }

    private static HybridResult[] Finish(Dictionary<int, Entry> merged, int limit)
    {
        if (merged.Count == 0 || limit <= 0)
        {
            return [];
        }

        var entries = new List<Entry>(merged.Values);
        entries.Sort(static (a, b) =>
        {
            int byScore = b.Score.CompareTo(a.Score);
            return byScore != 0 ? byScore : a.Id.CompareTo(b.Id);
        });

        int take = Math.Min(limit, entries.Count);
        var results = new HybridResult[take];
        for (int i = 0; i < take; i++)
        {
            Entry entry = entries[i];
            results[i] = new HybridResult
            {
                Id = entry.Id,
                Score = (float)entry.Score,
                VectorScore = entry.VectorScore,
                TextScore = entry.TextScore,
                VectorRank = entry.VectorRank,
                TextRank = entry.TextRank,
            };
        }

        return results;
    }
}
