using Faro.Filtering;
using Faro.Hybrid;

namespace Faro.Payload;

/// <summary>
/// In-memory payload records and their inverted indexes, keyed by the vector index's ids.
/// </summary>
internal sealed class PayloadStore<T>
    where T : class
{
    private readonly PayloadSchema<T> _schema;
    private readonly IPayloadSerializer<T> _serializer;
    private readonly Tokenizer _tokenizer;
    private readonly object _gate = new();
    private readonly Dictionary<string, IPayloadIndex> _indexes = new(StringComparer.Ordinal);
    private readonly Dictionary<int, PayloadRecord<T>> _records = new();
    private readonly HashSet<int> _allIds = [];

    internal PayloadStore(
        PayloadSchema<T> schema,
        IPayloadSerializer<T> serializer,
        Tokenizer tokenizer,
        Bm25Index textIndex)
    {
        _schema = schema;
        _serializer = serializer;
        _tokenizer = tokenizer;
        TextIndex = textIndex;

        foreach ((string name, PayloadSchema<T>.FieldDefinition definition) in schema.Fields)
        {
            IPayloadIndex? index = definition.Kind switch
            {
                FieldKind.Keyword or FieldKind.Tag => new TermIndex(),
                FieldKind.Numeric => new NumericIndex(),
                FieldKind.Bool => new BoolIndex(),
                FieldKind.Text => null, // text lives in the BM25 index, not in the filter indexes
                _ => throw new InvalidOperationException($"unsupported field kind {definition.Kind}"),
            };

            if (index is not null)
            {
                _indexes.Add(name, index);
            }
        }
    }

    internal Bm25Index TextIndex { get; }

    internal string? FullTextFieldName => _schema.FullTextFieldName;

    internal int Count
    {
        get { lock (_gate) { return _allIds.Count; } }
    }

    internal byte[] Serialize(T payload)
    {
        ArgumentNullException.ThrowIfNull(payload);
        return _serializer.Serialize(payload);
    }

    internal void Upsert(int id, T payload, byte[] serialized)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(id);
        ArgumentNullException.ThrowIfNull(payload);
        ArgumentNullException.ThrowIfNull(serialized);

        var values = new Dictionary<string, FieldValue>(_schema.Fields.Count, StringComparer.Ordinal);
        foreach ((string name, PayloadSchema<T>.FieldDefinition definition) in _schema.Fields)
        {
            values.Add(name, definition.Extractor(payload));
        }

        string? text = FullTextFieldName is { } field ? values[field].AsString() : null;

        lock (_gate)
        {
            if (_records.TryGetValue(id, out PayloadRecord<T>? previous))
            {
                RemoveFromIndexesLocked(id, previous.Values);
                TextIndex.Remove(id);
            }

            foreach ((string name, FieldValue value) in values)
            {
                if (_indexes.TryGetValue(name, out IPayloadIndex? index))
                {
                    switch (index)
                    {
                        case TermIndex term:
                            term.Add(id, value);
                            break;
                        case NumericIndex numeric:
                            numeric.Add(id, value.AsNumber());
                            break;
                        case BoolIndex boolean:
                            boolean.Add(id, value.AsBool());
                            break;
                    }
                }
            }

            if (text is not null && FullTextFieldName is not null)
            {
                TextIndex.AddOrReplace(id, text, _tokenizer);
            }

            _records[id] = new PayloadRecord<T>(id, payload, serialized, values);
            _allIds.Add(id);
        }
    }

    internal bool Delete(int id)
    {
        lock (_gate)
        {
            if (!_records.Remove(id, out PayloadRecord<T>? record))
            {
                return false;
            }

            RemoveFromIndexesLocked(id, record.Values);
            TextIndex.Remove(id);
            return _allIds.Remove(id);
        }
    }

    internal bool TryGetPayload(int id, out T? payload)
    {
        lock (_gate)
        {
            if (_records.TryGetValue(id, out PayloadRecord<T>? record))
            {
                payload = record.Payload;
                return true;
            }
        }

        payload = null;
        return false;
    }

    internal byte[]? GetSerialized(int id)
    {
        lock (_gate)
        {
            return _records.TryGetValue(id, out PayloadRecord<T>? record) ? record.Serialized : null;
        }
    }

    internal HashSet<int> Evaluate(Filter filter)
    {
        ArgumentNullException.ThrowIfNull(filter);
        lock (_gate)
        {
            return filter.Evaluate(CreateViewLocked());
        }
    }

    internal HashSet<int> AllIds()
    {
        lock (_gate)
        {
            return new HashSet<int>(_allIds);
        }
    }

    internal bool Matches(int id, Filter filter)
    {
        ArgumentNullException.ThrowIfNull(filter);
        lock (_gate)
        {
            return _records.TryGetValue(id, out PayloadRecord<T>? record) && filter.Matches(record);
        }
    }

    internal void ValidateFields(Filter filter)
    {
        var fields = new HashSet<string>(StringComparer.Ordinal);
        filter.CollectFields(fields);
        foreach (string field in fields)
        {
            if (!_schema.TryGetKind(field, out FieldKind kind) || kind == FieldKind.Text)
            {
                throw new ArgumentException(
                    $"filter field '{field}' is not an indexed payload field of this collection",
                    nameof(filter));
            }
        }
    }

    private void RemoveFromIndexesLocked(int id, IReadOnlyDictionary<string, FieldValue> values)
    {
        foreach ((string name, FieldValue value) in values)
        {
            if (_indexes.TryGetValue(name, out IPayloadIndex? index) && index is not null)
            {
                switch (index)
                {
                    case TermIndex term:
                        term.Remove(id, value);
                        break;
                    case NumericIndex numeric:
                        numeric.Remove(id, value.AsNumber());
                        break;
                    case BoolIndex boolean:
                        boolean.Remove(id, value.AsBool());
                        break;
                }
            }
        }
    }

    private FilterIndexView CreateViewLocked()
    {
        var view = new FilterIndexView(new HashSet<int>(_allIds));
        foreach ((string name, IPayloadIndex index) in _indexes)
        {
            view.Add(name, index);
        }

        return view;
    }

    private sealed class PayloadRecord<TValue>(int id, TValue payload, byte[] serialized, Dictionary<string, FieldValue> values)
        : IFieldValueLookup
        where TValue : class
    {
        internal int Id { get; } = id;

        internal TValue Payload { get; } = payload;

        internal byte[] Serialized { get; } = serialized;

        internal IReadOnlyDictionary<string, FieldValue> Values { get; } = values;

        public bool TryGet(string field, out FieldValue value) => Values.TryGetValue(field, out value);
    }
}
