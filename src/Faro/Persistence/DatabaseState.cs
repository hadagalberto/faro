namespace Faro.Persistence;

/// <summary>
/// Payload-agnostic in-memory mirror of everything that has been committed. The WAL appends
/// changes to this state; checkpoints serialize it to the main file.
/// </summary>
internal sealed class CollectionState
{
    private readonly object _gate = new();
    private readonly Dictionary<int, PersistedNode> _nodes = [];
    private readonly Dictionary<int, PersistedRecord> _records = [];
    private readonly Dictionary<string, int> _keyToId = new(StringComparer.Ordinal);

    internal CollectionState(CollectionDefinition definition)
    {
        Definition = definition;
    }

    internal CollectionDefinition Definition { get; }

    internal int NextId { get; private set; }

    internal PersistedNode[] SnapshotNodes()
    {
        lock (_gate)
        {
            var result = new PersistedNode[_nodes.Count];
            int i = 0;
            foreach (PersistedNode node in _nodes.Values)
            {
                result[i++] = node;
            }

            Array.Sort(result, static (a, b) => a.Id.CompareTo(b.Id));
            return result;
        }
    }

    internal PersistedRecord[] SnapshotRecords()
    {
        lock (_gate)
        {
            var result = new PersistedRecord[_records.Count];
            int i = 0;
            foreach (PersistedRecord record in _records.Values)
            {
                result[i++] = record;
            }

            Array.Sort(result, static (a, b) => a.InternalId.CompareTo(b.InternalId));
            return result;
        }
    }

    internal void ApplyUpsert(PersistedNode node, PersistedRecord record)
    {
        lock (_gate)
        {
            _nodes[node.Id] = node;
            _records[record.InternalId] = record;
            _keyToId[record.KeyToken] = record.InternalId;
            NextId = Math.Max(NextId, node.Id + 1);
        }
    }

    internal void ApplyDelete(int id)
    {
        lock (_gate)
        {
            if (_nodes.TryGetValue(id, out PersistedNode? node))
            {
                _nodes[id] = node.WithDeleted(true);
            }

            if (_records.Remove(id, out PersistedRecord? record))
            {
                _keyToId.Remove(record.KeyToken);
            }
        }
    }

    internal int? FindId(string keyToken)
    {
        lock (_gate)
        {
            return _keyToId.TryGetValue(keyToken, out int id) ? id : null;
        }
    }

    internal void ReplaceAll(PersistedNode[] nodes, PersistedRecord[] records)
    {
        lock (_gate)
        {
            _nodes.Clear();
            _records.Clear();
            _keyToId.Clear();
            NextId = 0;
            foreach (PersistedNode node in nodes)
            {
                _nodes[node.Id] = node;
                NextId = Math.Max(NextId, node.Id + 1);
            }

            foreach (PersistedRecord record in records)
            {
                _records[record.InternalId] = record;
                _keyToId[record.KeyToken] = record.InternalId;
            }
        }
    }
}

/// <summary>Thread-safe set of collections in one database.</summary>
internal sealed class DatabaseState
{
    private readonly object _gate = new();
    private readonly Dictionary<int, CollectionState> _byId = [];
    private readonly Dictionary<string, CollectionState> _byName = new(StringComparer.Ordinal);
    private int _nextCollectionId;

    internal int NextCollectionId
    {
        get { lock (_gate) { return _nextCollectionId; } }
    }

    internal CollectionState[] Snapshot()
    {
        lock (_gate)
        {
            var result = new CollectionState[_byId.Count];
            int i = 0;
            foreach (CollectionState collection in _byId.Values)
            {
                result[i++] = collection;
            }

            Array.Sort(result, static (a, b) => a.Definition.Id.CompareTo(b.Definition.Id));
            return result;
        }
    }

    internal CollectionState Add(CollectionDefinition definition)
    {
        lock (_gate)
        {
            if (_byName.ContainsKey(definition.Name))
            {
                throw new InvalidOperationException($"collection '{definition.Name}' already exists");
            }

            var state = new CollectionState(definition);
            _byId.Add(definition.Id, state);
            _byName.Add(definition.Name, state);
            _nextCollectionId = Math.Max(_nextCollectionId, definition.Id + 1);
            return state;
        }
    }

    internal bool TryGet(string name, out CollectionState? state)
    {
        lock (_gate)
        {
            return _byName.TryGetValue(name, out state);
        }
    }

    /// <summary>Removes a collection by id, returning its definition when it existed.</summary>
    internal CollectionDefinition? Remove(int id)
    {
        lock (_gate)
        {
            if (!_byId.TryGetValue(id, out CollectionState? state))
            {
                return null;
            }

            _byId.Remove(id);
            _byName.Remove(state.Definition.Name);
            return state.Definition;
        }
    }

    internal bool TryGet(int id, out CollectionState? state)
    {
        lock (_gate)
        {
            return _byId.TryGetValue(id, out state);
        }
    }

    /// <summary>Deterministic 64-bit FNV-1a over all collection definitions, stored in the header.</summary>
    internal ulong ComputeDatabaseId()
    {
        ulong hash = 14695981039346656037UL;
        foreach (CollectionState collection in Snapshot())
        {
            var writer = new ByteWriter();
            collection.Definition.Write(writer);
            foreach (byte value in writer.Span)
            {
                hash ^= value;
                hash *= 1099511628211UL;
            }
        }

        return hash;
    }
}
