using System.Collections.Concurrent;
using Faro.Filtering;
using Faro.Hybrid;
using Faro.Payload;
using Faro.Persistence;

namespace Faro;

/// <summary>
/// A collection in a <see cref="FaroDb"/> keyed by <see cref="int"/>.
/// </summary>
/// <typeparam name="TPayload">The payload type.</typeparam>
public sealed class FaroDbCollection<TPayload> : FaroDbCollection<int, TPayload>
    where TPayload : class
{
    internal FaroDbCollection(
        FaroDb db,
        CollectionState state,
        CollectionDefinition definition,
        PayloadSchema<TPayload> schema,
        IPayloadSerializer<TPayload> serializer,
        CollectionOptions<TPayload> options)
        : base(db, state, definition, schema, serializer, options)
    {
    }
}

/// <summary>
/// A durable vector collection with typed payloads, generic keys, filtered and hybrid search.
/// </summary>
/// <remarks>
/// Every mutation appends a self-contained WAL record, updates the in-memory indexes and updates
/// the database's checkpoint state. Keys are mapped to stable internal ids; updating an existing
/// key replaces its vector in place, so ids survive upserts. Concurrent batch upserts are safe:
/// HNSW inserts and payload updates are already fine-grained thread-safe, and WAL appends are
/// serialized by the log itself.
/// </remarks>
/// <typeparam name="TKey">One of <see cref="int"/>, <see cref="long"/>, <see cref="Guid"/> or <see cref="string"/>.</typeparam>
/// <typeparam name="TPayload">The payload type.</typeparam>
public class FaroDbCollection<TKey, TPayload> : IPersistedCollection
    where TKey : notnull
    where TPayload : class
{
    private readonly FaroDb _db;
    private readonly CollectionState _state;
    private readonly int _collectionId;
    private readonly IPayloadSerializer<TPayload> _serializer;
    private readonly FaroCollection<TPayload> _collection;
    private readonly ConcurrentDictionary<TKey, int> _keyToId;
    private readonly ConcurrentDictionary<int, TKey> _idToKey;

    internal FaroDbCollection(
        FaroDb db,
        CollectionState state,
        CollectionDefinition definition,
        PayloadSchema<TPayload> schema,
        IPayloadSerializer<TPayload> serializer,
        CollectionOptions<TPayload> options)
    {
        _db = db;
        _state = state;
        _collectionId = definition.Id;
        _serializer = serializer;
        _keyToId = new ConcurrentDictionary<TKey, int>();
        _idToKey = new ConcurrentDictionary<int, TKey>();

        var index = new HnswIndex(options.BuildIndexOptions());
        _collection = new FaroCollection<TPayload>(schema, serializer, index, options);

        index.Restore(state.SnapshotNodes(), db.MappedVectors, db.VectorOffsets(definition.Id));
        foreach (PersistedRecord record in state.SnapshotRecords())
        {
            TPayload? payload = serializer.Deserialize(record.Payload);
            if (payload is null)
            {
                throw new InvalidDataException(
                    $"persisted payload for internal id {record.InternalId} deserialized to null");
            }

            _collection.ReplacePayload(record.InternalId, payload, record.Payload);
            var reader = new ByteReader(record.Key);
            TKey key = KeyCodec<TKey>.Read(ref reader);
            _keyToId[key] = record.InternalId;
            _idToKey[record.InternalId] = key;
        }
    }

    /// <summary>Number of live records in the collection.</summary>
    public int Count => _collection.Count;

    /// <summary>Name of the collection.</summary>
    public string Name => _state.Definition.Name;

    /// <summary>The underlying vector index.</summary>
    public HnswIndex Index => _collection.Index;

    /// <summary>The BM25 index backing hybrid search.</summary>
    public Bm25Index TextIndex => _collection.TextIndex;

    /// <summary>All live keys, in unspecified order.</summary>
    public IReadOnlyCollection<TKey> Keys => _keyToId.Keys.ToArray();

    /// <summary>The planner decision made by the most recent search.</summary>
    public SearchDecision LastDecision => _collection.LastDecision;

    int IPersistedCollection.CollectionId => _collectionId;

    /// <summary>Inserts or replaces one record.</summary>
    /// <param name="key">The caller's stable key.</param>
    /// <param name="vector">Vector with the collection's dimensionality.</param>
    /// <param name="payload">The typed payload to store.</param>
    public void Upsert(TKey key, ReadOnlySpan<float> vector, TPayload payload) =>
        UpsertCore(key, vector, payload);

    /// <summary>
    /// Upserts many records. Payload preparation and HNSW insertion run in parallel; WAL appends
    /// are serialized by the log, so the resulting log is a valid sequence of independent records.
    /// </summary>
    /// <param name="items">Keys, vectors and payloads to upsert.</param>
    public void UpsertBatch(IEnumerable<(TKey Key, float[] Vector, TPayload Payload)> items)
    {
        ArgumentNullException.ThrowIfNull(items);
        (TKey Key, float[] Vector, TPayload Payload)[] all = items.ToArray();
        foreach ((TKey key, float[] vector, TPayload payload) in all)
        {
            ValidateArguments(key, vector, payload);
        }

        // The last occurrence of a repeated key wins; parallel workers never race on one key.
        var distinct = new Dictionary<TKey, (TKey Key, float[] Vector, TPayload Payload)>();
        for (int i = 0; i < all.Length; i++)
        {
            distinct[all[i].Key] = all[i];
        }

        Parallel.ForEach(distinct.Values, item => UpsertCore(item.Key, item.Vector, item.Payload));
    }

    /// <summary>Asynchronously inserts or replaces one record.</summary>
    /// <param name="key">The caller's stable key.</param>
    /// <param name="vector">Vector with the collection's dimensionality.</param>
    /// <param name="payload">The typed payload to store.</param>
    /// <param name="cancellationToken">Cancels the operation before it starts.</param>
    public Task UpsertAsync(
        TKey key, float[] vector, TPayload payload, CancellationToken cancellationToken = default) =>
        Task.Run(() => UpsertCore(key, vector, payload), cancellationToken);

    /// <summary>Asynchronously upserts a batch of records.</summary>
    /// <param name="items">Keys, vectors and payloads to upsert.</param>
    /// <param name="cancellationToken">Cancels the operation before it starts.</param>
    public Task UpsertBatchAsync(
        IEnumerable<(TKey Key, float[] Vector, TPayload Payload)> items,
        CancellationToken cancellationToken = default) =>
        Task.Run(() => UpsertBatch(items), cancellationToken);

    /// <summary>Deletes a record by key.</summary>
    /// <param name="key">The key to remove.</param>
    /// <returns><see langword="true"/> when a live record was removed.</returns>
    public bool Delete(TKey key)
    {
        ValidateKey(key);
        _db.Gate.EnterReadLock();
        try
        {
            if (!_keyToId.TryRemove(key, out int id))
            {
                return false;
            }

            _idToKey.TryRemove(id, out _);
            _collection.Delete(id);
            _state.ApplyDelete(id);
            _db.AppendDelete(_collectionId, id);
            return true;
        }
        finally
        {
            _db.Gate.ExitReadLock();
        }
    }

    /// <summary>Asynchronously deletes a record by key.</summary>
    /// <param name="key">The key to remove.</param>
    /// <param name="cancellationToken">Cancels the operation before it starts.</param>
    public Task<bool> DeleteAsync(TKey key, CancellationToken cancellationToken = default) =>
        Task.Run(() => Delete(key), cancellationToken);

    /// <summary>Tries to read the current payload for a key.</summary>
    /// <param name="key">The key to look up.</param>
    /// <param name="payload">The payload when found.</param>
    /// <returns><see langword="true"/> when the key exists.</returns>
    public bool TryGet(TKey key, out TPayload? payload)
    {
        ValidateKey(key);
        if (!_keyToId.TryGetValue(key, out int id))
        {
            payload = null;
            return false;
        }

        return _collection.TryGetPayload(id, out payload);
    }

    /// <summary>Whether the collection contains <paramref name="key"/>.</summary>
    /// <param name="key">The key to test.</param>
    public bool ContainsKey(TKey key)
    {
        ValidateKey(key);
        return _keyToId.ContainsKey(key);
    }

    /// <summary>Searches by vector, optionally restricted by a payload filter.</summary>
    /// <param name="query">The query vector.</param>
    /// <param name="k">Maximum number of hits.</param>
    /// <param name="filter">Optional payload filter.</param>
    /// <param name="efSearch">Optional beam width override.</param>
    /// <returns>Best-first hits with caller keys.</returns>
    public SearchHit<TKey>[] Search(
        ReadOnlySpan<float> query, int k, Filter? filter = null, int? efSearch = null)
    {
        SearchResult[] hits = _collection.Search(query, k, filter, efSearch);
        var result = new List<SearchHit<TKey>>(hits.Length);
        foreach (SearchResult hit in hits)
        {
            if (_idToKey.TryGetValue(hit.Id, out TKey? key))
            {
                result.Add(new SearchHit<TKey> { Key = key, Score = hit.Score });
            }
        }

        return result.ToArray();
    }

    /// <summary>Asynchronously searches by vector.</summary>
    /// <param name="query">The query vector.</param>
    /// <param name="k">Maximum number of hits.</param>
    /// <param name="filter">Optional payload filter.</param>
    /// <param name="efSearch">Optional beam width override.</param>
    /// <param name="cancellationToken">Cancels the operation before it starts.</param>
    public Task<SearchHit<TKey>[]> SearchAsync(
        float[] query,
        int k,
        Filter? filter = null,
        int? efSearch = null,
        CancellationToken cancellationToken = default) =>
        Task.Run(() => Search(query, k, filter, efSearch), cancellationToken);

    /// <summary>Runs a vector and BM25 search and fuses the two rankings.</summary>
    /// <param name="query">The query vector.</param>
    /// <param name="text">The full-text query.</param>
    /// <param name="k">Maximum number of fused hits.</param>
    /// <param name="fusion">Fusion strategy; defaults to RRF with k=60.</param>
    /// <param name="filter">Optional payload filter applied to both legs.</param>
    /// <param name="efSearch">Optional beam width override for the vector leg.</param>
    /// <returns>Fused hits, best first.</returns>
    public HybridHit<TKey>[] HybridSearch(
        ReadOnlySpan<float> query,
        string text,
        int k,
        Fusion? fusion = null,
        Filter? filter = null,
        int? efSearch = null)
    {
        HybridResult[] hits = _collection.HybridSearch(query, text, k, fusion, filter, efSearch);
        var result = new List<HybridHit<TKey>>(hits.Length);
        foreach (HybridResult hit in hits)
        {
            if (_idToKey.TryGetValue(hit.Id, out TKey? key))
            {
                result.Add(new HybridHit<TKey>
                {
                    Key = key,
                    Score = hit.Score,
                    VectorScore = hit.VectorScore,
                    TextScore = hit.TextScore,
                    VectorRank = hit.VectorRank,
                    TextRank = hit.TextRank,
                });
            }
        }

        return result.ToArray();
    }

    /// <summary>Asynchronously runs a hybrid search.</summary>
    /// <param name="query">The query vector.</param>
    /// <param name="text">The full-text query.</param>
    /// <param name="k">Maximum number of fused hits.</param>
    /// <param name="fusion">Fusion strategy.</param>
    /// <param name="filter">Optional payload filter.</param>
    /// <param name="efSearch">Optional beam width override.</param>
    /// <param name="cancellationToken">Cancels the operation before it starts.</param>
    public Task<HybridHit<TKey>[]> HybridSearchAsync(
        float[] query,
        string text,
        int k,
        Fusion? fusion = null,
        Filter? filter = null,
        int? efSearch = null,
        CancellationToken cancellationToken = default) =>
        Task.Run(() => HybridSearch(query, text, k, fusion, filter, efSearch), cancellationToken);

    /// <summary>Rebuilds the graph without tombstoned nodes.</summary>
    public void Compact() => _collection.Compact();

    Bm25Snapshot IPersistedCollection.GetTextSnapshot() => _collection.TextIndex.Export();

    void IPersistedCollection.Rebind(MappedVectorRegion region) =>
        _collection.Index.RebindMappedVectors(region);

    private void UpsertCore(TKey key, ReadOnlySpan<float> vector, TPayload payload)
    {
        ValidateArguments(key, vector, payload);
        _db.Gate.EnterReadLock();
        try
        {
            byte[] serialized = _serializer.Serialize(payload);
            int id;
            if (_keyToId.TryGetValue(key, out id) &&
                _collection.Index.IsAlive(id) &&
                _collection.Index.Replace(id, vector))
            {
                _collection.ReplacePayload(id, payload, serialized);
            }
            else
            {
                id = _collection.AddCore(vector, payload, serialized);
                _keyToId[key] = id;
                _idToKey[id] = key;
            }

            PersistedNode node = _collection.Index.ExportNode(id);
            var record = new PersistedRecord(id, EncodeKey(key), serialized);
            _state.ApplyUpsert(node, record);
            _db.AppendUpsert(_collectionId, node, record);
        }
        finally
        {
            _db.Gate.ExitReadLock();
        }
    }

    private byte[] EncodeKey(TKey key)
    {
        var writer = new ByteWriter();
        KeyCodec<TKey>.Write(key, writer);
        return writer.ToArray();
    }

    private static void ValidateArguments(TKey key, ReadOnlySpan<float> vector, TPayload payload)
    {
        ValidateKey(key);
        ArgumentNullException.ThrowIfNull(payload);
        if (vector.Length == 0)
        {
            throw new ArgumentException("vector must not be empty", nameof(vector));
        }
    }

    private static void ValidateKey(TKey key)
    {
        if (key is null)
        {
            throw new ArgumentNullException(nameof(key));
        }
    }
}
