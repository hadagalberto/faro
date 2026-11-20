namespace Faro.Payload;

/// <summary>
/// Declares which payload fields are indexed and how their values are extracted from a record.
/// </summary>
/// <remarks>
/// <para>
/// A schema is immutable once handed to a collection. It is built with delegate selectors rather
/// than reflection so it works under Native AOT and trimming, and so field names are explicit and
/// stable regardless of member renaming.
/// </para>
/// <para>
/// At most one <see cref="FullText(string, Func{T, string})"/> field may be declared; it feeds the
/// collection's BM25 index. Keyword, numeric, bool and tag fields feed the payload indexes used
/// by the filter planner.
/// </para>
/// </remarks>
/// <typeparam name="T">The payload type.</typeparam>
public sealed class PayloadSchema<T>
    where T : class
{
    private readonly Dictionary<string, FieldDefinition> _fields = new(StringComparer.Ordinal);

    /// <summary>Names of all declared fields, in registration order.</summary>
    public IReadOnlyCollection<string> FieldNames => _fields.Keys;

    internal IReadOnlyDictionary<string, FieldDefinition> Fields => _fields;

    internal string? FullTextFieldName { get; private set; }

    /// <summary>Returns the kind declared for <paramref name="name"/>.</summary>
    /// <param name="name">The field name.</param>
    /// <exception cref="ArgumentException">The field is not declared.</exception>
    public FieldKind KindOf(string name)
    {
        if (!_fields.TryGetValue(name, out FieldDefinition definition))
        {
            throw new ArgumentException($"payload field '{name}' is not declared in this schema", nameof(name));
        }

        return definition.Kind;
    }

    /// <summary>Tries to look up the declared kind of a field.</summary>
    /// <param name="name">The field name.</param>
    /// <param name="kind">The declared kind when found.</param>
    /// <returns><see langword="true"/> when the field is declared.</returns>
    public bool TryGetKind(string name, out FieldKind kind)
    {
        if (_fields.TryGetValue(name, out FieldDefinition definition))
        {
            kind = definition.Kind;
            return true;
        }

        kind = default;
        return false;
    }

    /// <summary>Declares a keyword (exact string equality) field.</summary>
    /// <param name="name">The field name used in filters.</param>
    /// <param name="selector">Extracts the field value from a payload.</param>
    /// <returns>This schema, for chaining.</returns>
    public PayloadSchema<T> Keyword(string name, Func<T, string?> selector) =>
        Add(name, FieldKind.Keyword, payload => FieldValue.String(selector(payload)));

    /// <summary>Declares a numeric field from a <see cref="double"/> selector.</summary>
    /// <param name="name">The field name used in filters.</param>
    /// <param name="selector">Extracts the field value from a payload.</param>
    /// <returns>This schema, for chaining.</returns>
    public PayloadSchema<T> Numeric(string name, Func<T, double> selector) =>
        Add(name, FieldKind.Numeric, payload => FieldValue.NumberValue(selector(payload)));

    /// <summary>Declares a numeric field from an <see cref="int"/> selector.</summary>
    /// <param name="name">The field name used in filters.</param>
    /// <param name="selector">Extracts the field value from a payload.</param>
    /// <returns>This schema, for chaining.</returns>
    public PayloadSchema<T> Numeric(string name, Func<T, int> selector) =>
        Add(name, FieldKind.Numeric, payload => FieldValue.NumberValue(selector(payload)));

    /// <summary>Declares a numeric field from a <see cref="long"/> selector.</summary>
    /// <param name="name">The field name used in filters.</param>
    /// <param name="selector">Extracts the field value from a payload.</param>
    /// <returns>This schema, for chaining.</returns>
    public PayloadSchema<T> Numeric(string name, Func<T, long> selector) =>
        Add(name, FieldKind.Numeric, payload => FieldValue.NumberValue(selector(payload)));

    /// <summary>Declares a boolean field.</summary>
    /// <param name="name">The field name used in filters.</param>
    /// <param name="selector">Extracts the field value from a payload.</param>
    /// <returns>This schema, for chaining.</returns>
    public PayloadSchema<T> Bool(string name, Func<T, bool> selector) =>
        Add(name, FieldKind.Bool, payload => FieldValue.Boolean(selector(payload)));

    /// <summary>Declares a tags field (any-of membership).</summary>
    /// <param name="name">The field name used in filters.</param>
    /// <param name="selector">Extracts the tags from a payload.</param>
    /// <returns>This schema, for chaining.</returns>
    public PayloadSchema<T> Tags(string name, Func<T, IEnumerable<string>?> selector) =>
        Add(name, FieldKind.Tag, payload => FieldValue.Tags(selector(payload)));

    /// <summary>Declares the full-text field fed into the BM25 index.</summary>
    /// <param name="name">The field name (used for diagnostics, not by filters).</param>
    /// <param name="selector">Extracts the text from a payload.</param>
    /// <returns>This schema, for chaining.</returns>
    /// <exception cref="InvalidOperationException">A full-text field was already declared.</exception>
    public PayloadSchema<T> FullText(string name, Func<T, string?> selector)
    {
        if (FullTextFieldName is not null)
        {
            throw new InvalidOperationException("only one full-text field may be declared per schema");
        }

        Add(name, FieldKind.Text, payload => FieldValue.String(selector(payload)));
        FullTextFieldName = name;
        return this;
    }

    internal FieldValue Extract(string name, T payload)
    {
        if (!_fields.TryGetValue(name, out FieldDefinition definition))
        {
            throw new ArgumentException($"payload field '{name}' is not declared in this schema", nameof(name));
        }

        return definition.Extractor(payload);
    }

    private PayloadSchema<T> Add(string name, FieldKind kind, Func<T, FieldValue> extractor)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(extractor);

        if (_fields.ContainsKey(name))
        {
            throw new ArgumentException($"payload field '{name}' is already declared", nameof(name));
        }

        _fields.Add(name, new FieldDefinition(kind, extractor));
        return this;
    }

    internal readonly record struct FieldDefinition(FieldKind Kind, Func<T, FieldValue> Extractor);
}
