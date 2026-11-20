using System.Buffers;
using Faro.Filtering;
using Faro.Hybrid;
using Faro.Internal;
using Faro.Payload;

namespace Faro;

/// <summary>
/// A vector collection with typed payloads, payload indexes, filtered search and hybrid search.
/// </summary>
/// <remarks>
/// <para>
/// The collection owns an <see cref="HnswIndex"/> plus the payload index set and BM25 inverted
/// index. Vector ids are the stable integers returned by <see cref="Add"/> and are shared by all
/// three structures, so a hit can be resolved back to its typed payload with
/// <see cref="TryGetPayload"/>.
/// </para>
/// <para>
/// Filtered searches are planned from payload-index selectivity: a very selective filter is scored
/// exactly over its candidate ids, while a broad filter traverses the graph with the predicate
/// applied at visit time and an adaptively widened beam. The decision for the last search is
/// exposed through <see cref="LastDecision"/>.
/// </para>
/// </remarks>
/// <typeparam name="T">The payload type.</typeparam>
public sealed class FaroCollection<T>
    where T : class
{
    private readonly HnswIndex _index;
    private readonly PayloadStore<T> _payloads;
    private SearchPlanner _planner;
    private readonly Tokenizer _tokenizer;
    private readonly int _defaultEf;
    private readonly int _hybridMultiplier;
    private readonly int _minHybridCandidates;

    /// <summary>Creates a collection over a new HNSW index.</summary>
    /// <param name="schema">The declared payload fields.</param>
    /// <param name="serializer">The AOT-safe payload serializer.</param>
    /// <param name="indexOptions">Options for the owned HNSW index.</param>
    /// <param name="options">Optional collection tuning.</param>
    /// <exception cref="ArgumentNullException">A required argument is <see langword="null"/>.</exception>
    public FaroCollection(
        PayloadSchema<T> schema,
        IPayloadSerializer<T> serializer,
        HnswOptions indexOptions,
        CollectionOptions? options = null)
        : this(schema, serializer, new HnswIndex(indexOptions), options)
    {
        // HnswIndex validates indexOptions (including null) before this body runs.
    }

    /// <summary>Creates a collection over an existing HNSW index.</summary>
    /// <param name="schema">The declared payload fields.</param>
    /// <param name="serializer">The AOT-safe payload serializer.</param>
    /// <param name="index">The vector index to own.</param>
    /// <param name="options">Optional collection tuning.</param>
    /// <exception cref="ArgumentNullException">A required argument is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">The schema declares no fields or no full-text field is available for hybrid search.</exception>
    public FaroCollection(
        PayloadSchema<T> schema,
        IPayloadSerializer<T> serializer,
        HnswIndex index,
        CollectionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(schema);
        ArgumentNullException.ThrowIfNull(serializer);
        ArgumentNullException.ThrowIfNull(index);

        if (schema.FieldNames.Count == 0)
        {
            throw new ArgumentException("the payload schema must declare at least one field", nameof(schema));
        }

        options ??= new CollectionOptions();
        if (options.HybridCandidateMultiplier < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "HybridCandidateMultiplier must be at least 1.");
        }

        if (options.MinHybridCandidates < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "MinHybridCandidates must be at least 1.");
        }

        if (options.DefaultEfSearch is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "DefaultEfSearch must be positive when supplied.");
        }

        _index = index;
        _planner = options.Planner ?? new SearchPlanner();
        _tokenizer = options.Tokenizer ?? Tokenizer.Default;
        _defaultEf = options.DefaultEfSearch ?? index.DefaultEfSearch;
        _hybridMultiplier = options.HybridCandidateMultiplier;
        _minHybridCandidates = options.MinHybridCandidates;

        var bm25 = new Bm25Index(options.Bm25K1 ?? 1.5, options.Bm25B ?? 0.75);
        _payloads = new PayloadStore<T>(schema, serializer, _tokenizer, bm25);
        LastDecision = new SearchDecision(
            SearchStrategy.FilteredTraversal,
            0,
            0,
            0,
            _defaultEf);
    }

    /// <summary>Number of live vectors in the collection.</summary>
    public int Count => _index.LiveCount;

    /// <summary>The underlying vector index.</summary>
    public HnswIndex Index => _index;

    /// <summary>The selectivity planner used by filtered searches. May be swapped for tuning.</summary>
    public SearchPlanner Planner
    {
        get => _planner;
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            _planner = value;
        }
    }

    /// <summary>The BM25 index backing hybrid search.</summary>
    public Bm25Index TextIndex => _payloads.TextIndex;

    /// <summary>The tokenizer used for the full-text field and hybrid queries.</summary>
    public Tokenizer Tokenizer => _tokenizer;

    /// <summary>The planner decision made by the most recent search.</summary>
    public SearchDecision LastDecision { get; private set; }

    /// <summary>Adds a vector and its typed payload, returning the new stable id.</summary>
    /// <param name="vector">The vector to add.</param>
    /// <param name="payload">The payload to index and serialize.</param>
    /// <returns>The id assigned to the new vector.</returns>
    public int Add(ReadOnlySpan<float> vector, T payload)
    {
        ArgumentNullException.ThrowIfNull(payload);
        // Serialize before touching the graph: a serializer failure must not leave an orphan
        // vector with no payload behind.
        byte[] serialized = _payloads.Serialize(payload);
        int id = _index.Add(vector);
        _payloads.Upsert(id, payload, serialized);
        return id;
    }

    /// <summary>Removes a vector and all its payload/text index entries.</summary>
    /// <param name="id">The id returned by <see cref="Add"/>.</param>
    /// <returns><see langword="true"/> when an existing live vector was removed.</returns>
    public bool Delete(int id)
    {
        bool removedPayload = _payloads.Delete(id);
        bool removedVector = _index.Delete(id);
        return removedPayload || removedVector;
    }

    /// <summary>Rebuilds the graph without tombstoned nodes.</summary>
    public void Compact() => _index.Compact();

    /// <summary>Deserializes the payload stored for an id.</summary>
    /// <param name="id">The vector id.</param>
    /// <param name="payload">The deserialized payload when found.</param>
    /// <returns><see langword="true"/> when the id has a payload.</returns>
    public bool TryGetPayload(int id, out T? payload) => _payloads.TryGetPayload(id, out payload);

    /// <summary>Searches by vector, optionally restricted by a payload filter.</summary>
    /// <param name="query">The query vector.</param>
    /// <param name="k">Maximum number of hits.</param>
    /// <param name="filter">The payload filter, or <see langword="null"/> for unfiltered search.</param>
    /// <param name="efSearch">Beam width override, or <see langword="null"/> for the default.</param>
    /// <returns>Best-first hits.</returns>
    /// <exception cref="ArgumentException">The filter references a field that is not indexed in this schema.</exception>
    public SearchResult[] Search(
        ReadOnlySpan<float> query,
        int k,
        Filter? filter = null,
        int? efSearch = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        int ef = efSearch ?? _defaultEf;
        if (ef <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(efSearch), "efSearch must be positive.");
        }

        if (filter is null || filter.IsAll)
        {
            int live = _index.LiveCount;
            LastDecision = new SearchDecision(SearchStrategy.FilteredTraversal, live, live, live == 0 ? 0 : 1, ef);
            return _index.Search(query, k, ef);
        }

        _payloads.ValidateFields(filter);
        HashSet<int> candidates = _payloads.Evaluate(filter);
        SearchDecision decision = _planner.Plan(candidates.Count, _index.LiveCount, ef);
        LastDecision = decision;

        if (candidates.Count == 0)
        {
            return [];
        }

        if (decision.Strategy == SearchStrategy.ExactCandidates)
        {
            return SearchExactCandidates(query, k, candidates);
        }

        return _index.SearchFiltered(
            query,
            k,
            id => _payloads.Matches(id, filter),
            decision.EffectiveEf,
            candidates);
    }

    /// <summary>Runs a vector and BM25 search and merges the two rankings.</summary>
    /// <param name="query">The query vector.</param>
    /// <param name="text">The full-text query.</param>
    /// <param name="k">Maximum number of fused hits.</param>
    /// <param name="fusion">Fusion strategy; defaults to reciprocal rank fusion with k=60.</param>
    /// <param name="filter">Optional payload filter applied to both legs.</param>
    /// <param name="efSearch">Beam width override for the vector leg.</param>
    /// <returns>Fused hits, best first.</returns>
    /// <exception cref="InvalidOperationException">The schema has no full-text field.</exception>
    public HybridResult[] HybridSearch(
        ReadOnlySpan<float> query,
        string text,
        int k,
        Fusion? fusion = null,
        Filter? filter = null,
        int? efSearch = null)
    {
        if (_payloads.FullTextFieldName is null)
        {
            throw new InvalidOperationException("hybrid search requires a full-text field in the payload schema");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        ArgumentNullException.ThrowIfNull(text);

        fusion ??= Fusion.Rrf();
        int candidateLimit = (int)Math.Min(int.MaxValue, Math.Max((long)k * _hybridMultiplier, _minHybridCandidates));

        HashSet<int> allowed;
        if (filter is null || filter.IsAll)
        {
            allowed = _payloads.AllIds();
        }
        else
        {
            _payloads.ValidateFields(filter);
            allowed = _payloads.Evaluate(filter);
        }

        if (allowed.Count == 0)
        {
            return [];
        }

        SearchResult[] vectorHits = Search(query, candidateLimit, filter, efSearch);
        var vectorLeg = new List<SearchResult>(vectorHits.Length);
        foreach (SearchResult hit in vectorHits)
        {
            if (allowed.Contains(hit.Id))
            {
                vectorLeg.Add(hit);
            }
        }

        (int DocId, double Score)[] textLeg = TextIndex.Search(
            text,
            _tokenizer,
            candidateLimit,
            allowed.Contains);

        return fusion.Fuse(vectorLeg, textLeg, Distances.LowerIsCloser(_index.Distance), k);
    }

    private SearchResult[] SearchExactCandidates(ReadOnlySpan<float> query, int k, IReadOnlyCollection<int> ids)
    {
        int limit = Math.Min(k, ids.Count);
        if (limit == 0)
        {
            return [];
        }

        var results = new SearchResult[limit];
        int dimensions = _index.Dimensions;
        bool hamming = _index.Distance == Distance.Hamming;

        float[]? normalized = null;
        ReadOnlySpan<float> prepared = query;
        if (_index.NormalizesOnAdd)
        {
            normalized = ArrayPool<float>.Shared.Rent(dimensions);
            Distances.CopyNormalized(query, normalized.AsSpan(0, dimensions));
            prepared = normalized.AsSpan(0, dimensions);
        }

        ulong[]? packedQuery = null;
        ulong[]? scratch = null;
        if (hamming)
        {
            int words = BitVectors.WordsFor(dimensions);
            packedQuery = ArrayPool<ulong>.Shared.Rent(words);
            scratch = ArrayPool<ulong>.Shared.Rent(words);
            BitVectors.PackSignBits(query, packedQuery.AsSpan(0, words));
        }

        float[] stored = ArrayPool<float>.Shared.Rent(dimensions);
        try
        {
            HeapOrder order = Distances.LowerIsCloser(_index.Distance)
                ? HeapOrder.MaxIdDescending
                : HeapOrder.MinIdDescending;
            using var heap = DistanceHeap.Rent(order, limit);

            foreach (int id in ids)
            {
                Span<float> destination = stored.AsSpan(0, dimensions);
                if (!_index.TryGetVector(id, destination))
                {
                    continue;
                }

                float score = VectorScoreHelper.Score(
                    _index.Distance,
                    _index.NormalizesOnAdd,
                    prepared,
                    packedQuery is null ? default : packedQuery.AsSpan(0, BitVectors.WordsFor(dimensions)),
                    destination,
                    scratch is null ? default : scratch.AsSpan(0, BitVectors.WordsFor(dimensions)));
                heap.PushBounded(limit, score, id);
            }

            int written = heap.Count;
            int index = written;
            while (heap.TryPop(out float score, out int id))
            {
                results[--index] = new SearchResult(id, score);
            }

            return written == results.Length ? results : results.AsSpan(0, written).ToArray();
        }
        finally
        {
            if (normalized is not null)
            {
                ArrayPool<float>.Shared.Return(normalized);
            }

            if (packedQuery is not null)
            {
                ArrayPool<ulong>.Shared.Return(packedQuery);
            }

            if (scratch is not null)
            {
                ArrayPool<ulong>.Shared.Return(scratch);
            }

            ArrayPool<float>.Shared.Return(stored);
        }
    }
}
