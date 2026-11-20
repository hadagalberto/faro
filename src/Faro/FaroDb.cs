using Faro.Hybrid;
using Faro.Payload;
using Faro.Persistence;

namespace Faro;

/// <summary>Implemented by collections that can contribute to a checkpoint.</summary>
internal interface IPersistedCollection
{
    int CollectionId { get; }
    Bm25Snapshot GetTextSnapshot();
    void Rebind(MappedVectorRegion region);
}

/// <summary>
/// An embedded database: one <c>.faro</c> file plus a write-ahead log, or a pure in-memory instance.
/// </summary>
/// <remarks>
/// Opening reads the checkpoint (if any), replays the WAL and truncates a torn tail. Mutations are
/// appended to the WAL before the process considers them durable; <see cref="Checkpoint"/> compacts
/// the log into a new file installed with an atomic rename. Vectors stored in the checkpoint are
/// read through a memory map, while vectors added since the last checkpoint live in arrays.
/// </remarks>
public sealed class FaroDb : IDisposable, IAsyncDisposable
{
    private readonly object _collectionGate = new();
    private readonly Dictionary<string, object> _collections = new(StringComparer.Ordinal);
    private readonly ReaderWriterLockSlim _gate = new(LockRecursionPolicy.SupportsRecursion);
    private readonly DatabaseState _state = new();
    private readonly HashSet<(int CollectionId, int NodeId)> _walUpdatedNodes = [];
    private Dictionary<int, Dictionary<int, long>> _vectorOffsets = [];
    private MappedVectorRegion? _mapped;
    private WalFile? _wal;
    private bool _disposed;

    private FaroDb(string? path, FaroOptions options, bool inMemory)
    {
        Path = path;
        IsInMemory = inMemory;
        Options = options;

        if (inMemory)
        {
            return;
        }

        CheckpointData checkpoint = Faro.Persistence.Checkpoint.Read(path!);
        foreach ((int id, CollectionDefinition definition) in checkpoint.Definitions)
        {
            _state.Add(definition);
            if (checkpoint.Nodes.TryGetValue(id, out PersistedNode[]? nodes) &&
                checkpoint.Records.TryGetValue(id, out PersistedRecord[]? records))
            {
                _state.TryGet(id, out CollectionState? state);
                state!.ReplaceAll(nodes, records);
            }
        }

        _wal = new WalFile(path! + WalFile.Suffix, options.Durability, options.TruncateTornTail);
        Replay(_wal.Replay);

        if (File.Exists(path!) && new FileInfo(path!).Length > 0)
        {
            _mapped = new MappedVectorRegion(path!);
            _vectorOffsets = Faro.Persistence.Checkpoint.ReadVectorOffsets(path!);
            foreach ((int collectionId, int nodeId) in _walUpdatedNodes)
            {
                if (_vectorOffsets.TryGetValue(collectionId, out Dictionary<int, long>? offsets))
                {
                    offsets.Remove(nodeId);
                }
            }
        }
    }

    /// <summary>Full path of the main database file, or <see langword="null"/> in in-memory mode.</summary>
    public string? Path { get; }

    /// <summary>Whether this database has no file and no WAL.</summary>
    public bool IsInMemory { get; }

    /// <summary>The options this database was opened with.</summary>
    public FaroOptions Options { get; }

    /// <summary>Names of all collections known to the database, including ones not yet materialised.</summary>
    public IReadOnlyCollection<string> CollectionNames
    {
        get
        {
            lock (_collectionGate)
            {
                return _collections.Keys.ToArray();
            }
        }
    }

    internal ReaderWriterLockSlim Gate => _gate;

    internal DatabaseState State => _state;

    internal MappedVectorRegion? MappedVectors => _mapped;

    /// <summary>Opens an existing database or creates a new one at <paramref name="path"/>.</summary>
    /// <param name="path">Path to the <c>.faro</c> file. Its WAL lives at <c>path + ".wal"</c>.</param>
    /// <param name="options">Durability and recovery options.</param>
    public static FaroDb Open(string path, FaroOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return new FaroDb(System.IO.Path.GetFullPath(path), options ?? new FaroOptions(), inMemory: false);
    }

    /// <summary>Creates a database that keeps all state in memory, with no file and no WAL.</summary>
    /// <param name="options">Options; durability settings have no effect in this mode.</param>
    public static FaroDb InMemory(FaroOptions? options = null) =>
        new(null, options ?? new FaroOptions(), inMemory: true);

    /// <summary>Gets or creates a collection keyed by <see cref="int"/>.</summary>
    /// <typeparam name="TPayload">The payload type.</typeparam>
    /// <param name="name">The collection name.</param>
    /// <param name="options">Collection definition and tuning.</param>
    public FaroDbCollection<TPayload> GetOrCreateCollection<TPayload>(
        string name,
        CollectionOptions<TPayload> options)
        where TPayload : class =>
        (FaroDbCollection<TPayload>)GetOrCreateCollection<int, TPayload>(name, options);

    /// <summary>Gets a previously materialised collection keyed by <see cref="int"/>.</summary>
    /// <typeparam name="TPayload">The payload type.</typeparam>
    /// <param name="name">The collection name.</param>
    /// <returns>The collection, or <see langword="null"/> when it has not been materialised.</returns>
    public FaroDbCollection<TPayload>? GetCollection<TPayload>(string name)
        where TPayload : class =>
        GetCollection<int, TPayload>(name) as FaroDbCollection<TPayload>;

    /// <summary>Gets or creates a collection with a caller-chosen key type.</summary>
    /// <typeparam name="TKey">One of <see cref="int"/>, <see cref="long"/>, <see cref="Guid"/> or <see cref="string"/>.</typeparam>
    /// <typeparam name="TPayload">The payload type.</typeparam>
    /// <param name="name">The collection name.</param>
    /// <param name="options">Collection definition and tuning.</param>
    /// <exception cref="NotSupportedException"><typeparamref name="TKey"/> is not a supported key type.</exception>
    /// <exception cref="InvalidOperationException">An existing collection has an incompatible definition.</exception>
    public FaroDbCollection<TKey, TPayload> GetOrCreateCollection<TKey, TPayload>(
        string name,
        CollectionOptions<TPayload> options)
        where TKey : notnull
        where TPayload : class
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(options);
        ThrowIfDisposed();
        KeyCodec<TKey>.Validate();

        lock (_collectionGate)
        {
            if (_collections.TryGetValue(name, out object? existing))
            {
                ValidateRequestedDefinition<TKey, TPayload>(name, options);
                return existing as FaroDbCollection<TKey, TPayload>
                    ?? throw new InvalidOperationException(
                        $"collection '{name}' is already open with a different key or payload type");
            }

            _gate.EnterWriteLock();
            try
            {
                PayloadSchema<TPayload> schema = options.BuildSchema();
                IPayloadSerializer<TPayload> serializer = options.ResolveSerializer();
                HnswOptions indexOptions = options.BuildIndexOptions();
                var fields = new StoredField[schema.Fields.Count];
                int i = 0;
                foreach ((string fieldName, PayloadSchema<TPayload>.FieldDefinition fieldDefinition) in schema.Fields)
                {
                    fields[i++] = new StoredField(fieldName, fieldDefinition.Kind);
                }

                CollectionDefinition? definitionFromDisk = null;
                if (_state.TryGet(name, out CollectionState? existingState))
                {
                    definitionFromDisk = existingState!.Definition;
                    ValidateCompatibility<TKey, TPayload>(definitionFromDisk, options, fields);
                }

                CollectionDefinition definition = definitionFromDisk ?? new CollectionDefinition(
                    _state.NextCollectionId,
                    name,
                    KeyCodec<TKey>.Kind,
                    options.Dimensions,
                    options.Distance,
                    indexOptions.NormalizeOnAdd ?? (options.Distance == Distance.Cosine),
                    indexOptions.M,
                    indexOptions.EfConstruction,
                    indexOptions.EfSearch,
                    indexOptions.Seed,
                    indexOptions.LevelMultiplier,
                    indexOptions.ExtendCandidates,
                    indexOptions.KeepPrunedConnections,
                    indexOptions.Mc,
                    options.Quantization?.Kind,
                    options.Quantization?.RescoreFactor ?? 4,
                    options.Quantization?.Quantile,
                    options.Bm25K1 ?? 1.5,
                    options.Bm25B ?? 0.75,
                    fields,
                    schema.FullTextFieldName);

                if (definitionFromDisk is null)
                {
                    _state.Add(definition);
                    AppendCreate(definition);
                }

                _state.TryGet(name, out CollectionState? collectionState);
                FaroDbCollection<TKey, TPayload> collection = typeof(TKey) == typeof(int)
                    ? (FaroDbCollection<TKey, TPayload>)(object)new FaroDbCollection<TPayload>(
                        this, collectionState!, definition, schema, serializer, options)
                    : new FaroDbCollection<TKey, TPayload>(
                        this, collectionState!, definition, schema, serializer, options);
                _collections.Add(name, collection);
                return collection;
            }
            finally
            {
                _gate.ExitWriteLock();
            }
        }
    }

    /// <summary>Gets a previously materialised collection with a caller-chosen key type.</summary>
    /// <typeparam name="TKey">The collection's key type.</typeparam>
    /// <typeparam name="TPayload">The payload type.</typeparam>
    /// <param name="name">The collection name.</param>
    /// <returns>The collection, or <see langword="null"/> when it has not been materialised.</returns>
    public FaroDbCollection<TKey, TPayload>? GetCollection<TKey, TPayload>(string name)
        where TKey : notnull
        where TPayload : class
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ThrowIfDisposed();
        lock (_collectionGate)
        {
            return _collections.TryGetValue(name, out object? value)
                ? value as FaroDbCollection<TKey, TPayload>
                : null;
        }
    }

    /// <summary>Flushes pending WAL records according to the configured durability mode.</summary>
    public void Flush()
    {
        ThrowIfDisposed();
        _wal?.Flush();
    }

    /// <summary>Asynchronously flushes pending WAL records.</summary>
    /// <param name="cancellationToken">Cancels the wait before the flush starts.</param>
    public Task FlushAsync(CancellationToken cancellationToken = default) =>
        Task.Run(Flush, cancellationToken);

    /// <summary>
    /// Writes a fresh checkpoint, atomically replaces the main file and resets the WAL.
    /// Not available in in-memory mode. Do not search while checkpointing.
    /// </summary>
    /// <exception cref="InvalidOperationException">The database is in-memory.</exception>
    public void Checkpoint()
    {
        ThrowIfDisposed();
        if (IsInMemory)
        {
            throw new InvalidOperationException("an in-memory database has no checkpoint file");
        }

        // Lock order must match GetOrCreateCollection: collection registry first, then the
        // read/write gate that excludes in-flight mutations.
        lock (_collectionGate)
        {
            _gate.EnterWriteLock();
            try
            {
                _wal!.Flush();
                var text = new Dictionary<int, Bm25Snapshot>();
                var collections = new List<IPersistedCollection>();
                foreach (object value in _collections.Values)
                {
                    if (value is IPersistedCollection persisted)
                    {
                        collections.Add(persisted);
                        text[persisted.CollectionId] = persisted.GetTextSnapshot();
                    }
                }

                ulong databaseId = _state.ComputeDatabaseId();
                _mapped?.Dispose();
                _mapped = null;

                Faro.Persistence.Checkpoint.Write(Path!, databaseId, _state.Snapshot(), text);
                _wal.ResetAfterCheckpoint();

                _mapped = new MappedVectorRegion(Path!);
                _vectorOffsets = Faro.Persistence.Checkpoint.ReadVectorOffsets(Path!);
                foreach (IPersistedCollection collection in collections)
                {
                    collection.Rebind(_mapped);
                }
            }
            finally
            {
                _gate.ExitWriteLock();
            }
        }
    }

    /// <summary>Asynchronously checkpoints the database.</summary>
    /// <param name="cancellationToken">Cancels the wait before the checkpoint starts.</param>
    public Task CheckpointAsync(CancellationToken cancellationToken = default) =>
        Task.Run(Checkpoint, cancellationToken);

    internal IReadOnlyDictionary<int, long> VectorOffsets(int collectionId) =>
        _vectorOffsets.TryGetValue(collectionId, out Dictionary<int, long>? offsets)
            ? offsets
            : new Dictionary<int, long>();

    internal void AppendUpsert(int collectionId, PersistedNode node, PersistedRecord record)
    {
        if (_wal is null)
        {
            return;
        }

        var payload = new ByteWriter();
        payload.WriteInt32(collectionId);
        node.Write(payload);
        record.Write(payload);
        _wal.Append(2, payload);
    }

    internal void AppendDelete(int collectionId, int internalId)
    {
        if (_wal is null)
        {
            return;
        }

        var payload = new ByteWriter();
        payload.WriteInt32(collectionId);
        payload.WriteInt32(internalId);
        _wal.Append(3, payload);
    }

    private void AppendCreate(CollectionDefinition definition)
    {
        if (_wal is null)
        {
            return;
        }

        var payload = new ByteWriter();
        payload.WriteInt32(definition.Id);
        definition.Write(payload);
        _wal.Append(1, payload);
    }

    private void Replay(WalReplay replay)
    {
        foreach (WalRecord record in replay.Records)
        {
            var reader = new ByteReader(record.Payload);
            try
            {
                switch (record.Type)
                {
                    case 1:
                    {
                        int id = reader.ReadInt32();
                        CollectionDefinition definition = CollectionDefinition.Read(reader);
                        if (!_state.TryGet(id, out _))
                        {
                            _state.Add(definition);
                        }

                        break;
                    }

                    case 2:
                    {
                        int collectionId = reader.ReadInt32();
                        PersistedNode node = PersistedNode.Read(reader);
                        PersistedRecord persisted = PersistedRecord.Read(reader);
                        if (_state.TryGet(collectionId, out CollectionState? state))
                        {
                            state!.ApplyUpsert(node, persisted);
                            _walUpdatedNodes.Add((collectionId, node.Id));
                        }

                        break;
                    }

                    case 3:
                    {
                        int collectionId = reader.ReadInt32();
                        int nodeId = reader.ReadInt32();
                        if (_state.TryGet(collectionId, out CollectionState? state))
                        {
                            state!.ApplyDelete(nodeId);
                        }

                        break;
                    }

                    default:
                        // Unknown record types are ignored; their CRC was already verified.
                        break;
                }
            }
            catch (InvalidDataException)
            {
                // A structurally invalid record behaves like a torn tail record.
            }
        }
    }

    private void ValidateRequestedDefinition<TKey, TPayload>(string name, CollectionOptions<TPayload> options)
        where TKey : notnull
        where TPayload : class
    {
        PayloadSchema<TPayload> schema = options.BuildSchema();
        var fields = new StoredField[schema.Fields.Count];
        int i = 0;
        foreach ((string fieldName, PayloadSchema<TPayload>.FieldDefinition fieldDefinition) in schema.Fields)
        {
            fields[i++] = new StoredField(fieldName, fieldDefinition.Kind);
        }

        if (_state.TryGet(name, out CollectionState? state))
        {
            ValidateCompatibility<TKey, TPayload>(state!.Definition, options, fields);
        }
    }

    private static void ValidateCompatibility<TKey, TPayload>(
        CollectionDefinition definition,
        CollectionOptions<TPayload> options,
        StoredField[] fields)
        where TKey : notnull
        where TPayload : class
    {
        bool sameFields = definition.Fields.Length == fields.Length &&
                          definition.Fields.Zip(fields).All(pair =>
                              pair.First.Name == pair.Second.Name && pair.First.Kind == pair.Second.Kind);
        string? expectedFullText = options.Schema is null
            ? options.FullTextField is null ? null : "text"
            : options.Schema.FullTextFieldName;
        if (definition.KeyKind != KeyCodec<TKey>.Kind ||
            definition.Dimensions != options.Dimensions ||
            definition.Distance != options.Distance ||
            !sameFields ||
            definition.FullTextField != expectedFullText)
        {
            throw new InvalidOperationException(
                $"collection '{definition.Name}' was created with a different key type, dimensionality, metric or schema");
        }
    }

    /// <summary>Flushes the WAL and releases the log and memory map.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _gate.EnterWriteLock();
        try
        {
            _wal?.Dispose();
            _wal = null;
            _mapped?.Dispose();
            _mapped = null;
            _disposed = true;
        }
        finally
        {
            _gate.ExitWriteLock();
        }
    }

    /// <summary>Asynchronously flushes and releases all file resources.</summary>
    public async ValueTask DisposeAsync()
    {
        if (!_disposed)
        {
            await Task.Run(Dispose).ConfigureAwait(false);
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
}
