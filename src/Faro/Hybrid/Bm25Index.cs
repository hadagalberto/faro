using System.Globalization;

namespace Faro.Hybrid;

/// <summary>
/// In-house BM25 inverted index (Robertson &amp; Zaragoza, 2009).
/// </summary>
/// <remarks>
/// Scores use the standard formulation
/// <c>idf(t) = ln(1 + (N - df + 0.5)/(df + 0.5))</c> and
/// <c>tf' = tf*(k1+1)/(tf + k1*(1 - b + b*dl/avgdl))</c>, summed over the query terms present in
/// the document. Only documents containing at least one query term are scored, so a query walks
/// postings lists rather than the whole corpus. Documents are identified by the same integer ids
/// as the vector index.
/// </remarks>
public sealed class Bm25Index
{
    private readonly double _k1;
    private readonly double _b;
    private readonly object _gate = new();

    private readonly Dictionary<string, Dictionary<int, int>> _postings = new(StringComparer.Ordinal);
    private readonly Dictionary<int, Dictionary<string, int>> _docTerms = new();
    private readonly Dictionary<int, int> _docLengths = new();

    /// <summary>Creates an empty BM25 index.</summary>
    /// <param name="k1">Term-frequency saturation; must be positive.</param>
    /// <param name="b">Document-length normalization, in [0, 1].</param>
    /// <exception cref="ArgumentOutOfRangeException">A parameter is outside its valid range.</exception>
    public Bm25Index(double k1 = 1.5, double b = 0.75)
    {
        if (double.IsNaN(k1) || k1 <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k1), k1, "k1 must be positive.");
        }

        if (double.IsNaN(b) || b < 0 || b > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(b), b, "b must be in [0, 1].");
        }

        _k1 = k1;
        _b = b;
    }

    /// <summary>Number of indexed documents.</summary>
    public int DocumentCount
    {
        get { lock (_gate) { return _docLengths.Count; } }
    }

    /// <summary>Number of distinct indexed terms.</summary>
    public int VocabularyCount
    {
        get { lock (_gate) { return _postings.Count; } }
    }

    /// <summary>Adds a document, replacing any previous text stored under the same id.</summary>
    /// <param name="docId">The document id.</param>
    /// <param name="text">The document text.</param>
    /// <param name="tokenizer">The tokenizer used for this document.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="docId"/> is negative.</exception>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> or <paramref name="tokenizer"/> is <see langword="null"/>.</exception>
    public void AddOrReplace(int docId, string text, Tokenizer tokenizer)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(docId);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(tokenizer);

        string[] tokens = tokenizer.Tokenize(text);
        lock (_gate)
        {
            RemoveLocked(docId);
            _docLengths.Add(docId, tokens.Length);

            var termFrequencies = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (string token in tokens)
            {
                termFrequencies.TryGetValue(token, out int frequency);
                termFrequencies[token] = frequency + 1;

                if (!_postings.TryGetValue(token, out Dictionary<int, int>? postings))
                {
                    postings = [];
                    _postings.Add(token, postings);
                }

                postings[docId] = frequency + 1;
            }

            _docTerms.Add(docId, termFrequencies);
        }
    }

    /// <summary>Removes a document from the index.</summary>
    /// <param name="docId">The document id.</param>
    /// <returns><see langword="true"/> when a document was removed.</returns>
    public bool Remove(int docId)
    {
        lock (_gate)
        {
            return RemoveLocked(docId);
        }
    }

    /// <summary>Gets the stored term frequency of one term in one document.</summary>
    /// <param name="docId">The document id.</param>
    /// <param name="token">The normalized term.</param>
    /// <returns>The term frequency, or zero when the term or document is absent.</returns>
    public int TermFrequency(int docId, string token)
    {
        ArgumentNullException.ThrowIfNull(token);
        lock (_gate)
        {
            return _docTerms.TryGetValue(docId, out Dictionary<string, int>? terms)
                && terms.TryGetValue(token, out int frequency)
                ? frequency
                : 0;
        }
    }

    /// <summary>Gets the token count stored for a document.</summary>
    /// <param name="docId">The document id.</param>
    /// <returns>The document length, or zero when the id is absent.</returns>
    public int DocumentLength(int docId)
    {
        lock (_gate)
        {
            return _docLengths.TryGetValue(docId, out int length) ? length : 0;
        }
    }

    /// <summary>Computes the BM25 score of one document for a query.</summary>
    /// <param name="query">The raw query text.</param>
    /// <param name="tokenizer">The tokenizer used for the query.</param>
    /// <param name="docId">The document id.</param>
    /// <returns>The BM25 score; zero when the document is absent or no term matches.</returns>
    public double Score(string query, Tokenizer tokenizer, int docId)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(tokenizer);

        string[] tokens = tokenizer.Tokenize(query);
        if (tokens.Length == 0)
        {
            return 0;
        }

        lock (_gate)
        {
            return ScoreLocked(tokens, docId, _docLengths.Count, AverageLengthLocked());
        }
    }

    /// <summary>Ranks documents by BM25 score for a query.</summary>
    /// <param name="query">The raw query text.</param>
    /// <param name="tokenizer">The tokenizer used for the query.</param>
    /// <param name="limit">Maximum number of results.</param>
    /// <param name="include">Optional predicate restricting which document ids may be returned.</param>
    /// <returns>Best-first <c>(docId, score)</c> pairs; ties are broken by ascending id.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="limit"/> is not positive.</exception>
    public (int DocId, double Score)[] Search(
        string query,
        Tokenizer tokenizer,
        int limit,
        Func<int, bool>? include = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(limit);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(tokenizer);

        string[] tokens = tokenizer.Tokenize(query);
        if (tokens.Length == 0)
        {
            return [];
        }

        lock (_gate)
        {
            var candidates = new HashSet<int>();
            var distinct = new HashSet<string>(tokens, StringComparer.Ordinal);
            foreach (string token in distinct)
            {
                if (_postings.TryGetValue(token, out Dictionary<int, int>? postings))
                {
                    candidates.UnionWith(postings.Keys);
                }
            }

            var scored = new List<(int DocId, double Score)>(candidates.Count);
            int docCount = _docLengths.Count;
            double averageLength = AverageLengthLocked();
            foreach (int docId in candidates)
            {
                if (include is not null && !include(docId))
                {
                    continue;
                }

                double score = ScoreLocked(tokens, docId, docCount, averageLength);
                if (score > 0)
                {
                    scored.Add((docId, score));
                }
            }

            scored.Sort(static (a, b) =>
            {
                int byScore = b.Score.CompareTo(a.Score);
                return byScore != 0 ? byScore : a.DocId.CompareTo(b.DocId);
            });

            return scored.Count <= limit
                ? scored.ToArray()
                : scored.GetRange(0, limit).ToArray();
        }
    }

    /// <summary>Exports postings and document lengths for the checkpoint text page.</summary>
    internal Bm25Snapshot Export()
    {
        lock (_gate)
        {
            var documents = new Bm25Snapshot.DocumentSnapshot[_docLengths.Count];
            int index = 0;
            foreach ((int docId, int length) in _docLengths)
            {
                var terms = new (string, int)[_docTerms[docId].Count];
                int t = 0;
                foreach ((string term, int frequency) in _docTerms[docId])
                {
                    terms[t++] = (term, frequency);
                }

                documents[index++] = new Bm25Snapshot.DocumentSnapshot(docId, length, terms);
            }

            Array.Sort(documents, static (a, b) => a.DocId.CompareTo(b.DocId));
            return new Bm25Snapshot(documents);
        }
    }

    /// <summary>Renders the index statistics, useful for diagnostics.</summary>
    /// <returns>A short human-readable summary.</returns>
    public override string ToString()
    {
        lock (_gate)
        {
            return string.Format(
                CultureInfo.InvariantCulture,
                "BM25(k1={0}, b={1}, docs={2}, terms={3})",
                _k1,
                _b,
                _docLengths.Count,
                _postings.Count);
        }
    }

    private bool RemoveLocked(int docId)
    {
        if (!_docTerms.TryGetValue(docId, out Dictionary<string, int>? terms))
        {
            return false;
        }

        foreach (string token in terms.Keys)
        {
            if (_postings.TryGetValue(token, out Dictionary<int, int>? postings))
            {
                postings.Remove(docId);
                if (postings.Count == 0)
                {
                    _postings.Remove(token);
                }
            }
        }

        _docTerms.Remove(docId);
        _docLengths.Remove(docId);
        return true;
    }

    private double AverageLengthLocked()
    {
        if (_docLengths.Count == 0)
        {
            return 0;
        }

        long total = 0;
        foreach (int length in _docLengths.Values)
        {
            total += length;
        }

        return total / (double)_docLengths.Count;
    }

    private double ScoreLocked(string[] queryTokens, int docId, int docCount, double avgLength)
    {
        if (!_docLengths.TryGetValue(docId, out int length) ||
            !_docTerms.TryGetValue(docId, out Dictionary<string, int>? terms))
        {
            return 0;
        }

        if (docCount == 0 || length == 0 || avgLength == 0)
        {
            return 0;
        }

        int n = docCount;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        double score = 0;

        foreach (string token in queryTokens)
        {
            if (!seen.Add(token) || !terms.TryGetValue(token, out int tf))
            {
                continue;
            }

            int df = _postings.TryGetValue(token, out Dictionary<int, int>? postings) ? postings.Count : 0;
            if (df == 0)
            {
                continue;
            }

            double idf = Math.Log(1.0 + (n - df + 0.5) / (df + 0.5));
            double denominator = tf + _k1 * (1.0 - _b + _b * length / avgLength);
            score += idf * (tf * (_k1 + 1.0) / denominator);
        }

        return score;
    }
}
