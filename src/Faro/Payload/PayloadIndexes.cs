namespace Faro.Payload;

/// <summary>Marker for an inverted index over one payload field.</summary>
internal interface IPayloadIndex
{
}

/// <summary>
/// Inverted index for keyword and tag fields: value to the set of vector ids carrying it.
/// </summary>
internal sealed class TermIndex : IPayloadIndex
{
    private readonly Dictionary<string, HashSet<int>> _postings = new(StringComparer.Ordinal);

    internal void Add(int id, FieldValue value)
    {
        if (value.Kind == FieldKind.Tag)
        {
            foreach (string tag in value.AsTags())
            {
                AddValue(tag, id);
            }
        }
        else
        {
            string text = value.AsString();
            if (value.Kind == FieldKind.Keyword && text.Length != 0)
            {
                AddValue(text, id);
            }
        }
    }

    internal void Remove(int id, FieldValue value)
    {
        if (value.Kind == FieldKind.Tag)
        {
            foreach (string tag in value.AsTags())
            {
                RemoveValue(tag, id);
            }
        }
        else if (value.Kind == FieldKind.Keyword)
        {
            RemoveValue(value.AsString(), id);
        }
    }

    internal HashSet<int> Get(string value) =>
        _postings.TryGetValue(value, out HashSet<int>? ids) ? new HashSet<int>(ids) : [];

    internal HashSet<int> GetAny(IEnumerable<string> values)
    {
        var result = new HashSet<int>();
        foreach (string value in values)
        {
            if (_postings.TryGetValue(value, out HashSet<int>? ids))
            {
                result.UnionWith(ids);
            }
        }

        return result;
    }

    private void AddValue(string value, int id)
    {
        if (!_postings.TryGetValue(value, out HashSet<int>? ids))
        {
            ids = [];
            _postings.Add(value, ids);
        }

        ids.Add(id);
    }

    private void RemoveValue(string value, int id)
    {
        if (_postings.TryGetValue(value, out HashSet<int>? ids))
        {
            ids.Remove(id);
            if (ids.Count == 0)
            {
                _postings.Remove(value);
            }
        }
    }
}

/// <summary>
/// Ordered index for numeric fields. Keys are kept sorted so a range scan stops as soon as the
/// upper bound is passed instead of visiting the whole vocabulary.
/// </summary>
internal sealed class NumericIndex : IPayloadIndex
{
    private readonly SortedDictionary<double, HashSet<int>> _values = new();

    internal void Add(int id, double value)
    {
        // NaN is never equal to or ordered against anything, so it cannot answer a filter.
        if (double.IsNaN(value))
        {
            return;
        }

        if (!_values.TryGetValue(value, out HashSet<int>? ids))
        {
            ids = [];
            _values.Add(value, ids);
        }

        ids.Add(id);
    }

    internal void Remove(int id, double value)
    {
        if (double.IsNaN(value) || !_values.TryGetValue(value, out HashSet<int>? ids))
        {
            return;
        }

        ids.Remove(id);
        if (ids.Count == 0)
        {
            _values.Remove(value);
        }
    }

    internal HashSet<int> Get(double value) =>
        !double.IsNaN(value) && _values.TryGetValue(value, out HashSet<int>? ids)
            ? new HashSet<int>(ids)
            : [];

    internal HashSet<int> Range(double min, double max, bool includeMin, bool includeMax)
    {
        var result = new HashSet<int>();
        foreach ((double value, HashSet<int> ids) in _values)
        {
            bool aboveMin = includeMin ? value >= min : value > min;
            if (!aboveMin)
            {
                continue;
            }

            bool belowMax = includeMax ? value <= max : value < max;
            if (!belowMax)
            {
                break;
            }

            result.UnionWith(ids);
        }

        return result;
    }
}

/// <summary>Inverted index for boolean fields.</summary>
internal sealed class BoolIndex : IPayloadIndex
{
    private readonly HashSet<int> _true = [];
    private readonly HashSet<int> _false = [];

    internal void Add(int id, bool value) => (value ? _true : _false).Add(id);

    internal void Remove(int id, bool value) => (value ? _true : _false).Remove(id);

    internal HashSet<int> Get(bool value) => new(value ? _true : _false);
}

/// <summary>
/// Snapshot of the payload indexes handed to a filter for candidate-set evaluation.
/// </summary>
internal sealed class FilterIndexView
{
    private readonly Dictionary<string, IPayloadIndex> _indexes = new(StringComparer.Ordinal);

    internal FilterIndexView(HashSet<int> allIds)
    {
        AllIds = allIds;
    }

    internal HashSet<int> AllIds { get; }

    internal void Add(string field, IPayloadIndex index) => _indexes.Add(field, index);

    internal TermIndex Term(string field) => Get<TermIndex>(field, FieldKind.Keyword);

    internal NumericIndex Numeric(string field) => Get<NumericIndex>(field, FieldKind.Numeric);

    internal BoolIndex Bool(string field) => Get<BoolIndex>(field, FieldKind.Bool);

    private TIndex Get<TIndex>(string field, FieldKind expected)
        where TIndex : IPayloadIndex
    {
        if (!_indexes.TryGetValue(field, out IPayloadIndex? index))
        {
            throw new ArgumentException($"payload field '{field}' is not declared in this schema", nameof(field));
        }

        if (index is not TIndex typed)
        {
            throw new ArgumentException(
                $"payload field '{field}' has kind {index.GetType().Name}; expected {expected} for this filter",
                nameof(field));
        }

        return typed;
    }
}
