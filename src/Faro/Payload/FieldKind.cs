namespace Faro.Payload;

/// <summary>
/// The indexed behaviour of a payload field declared in a <see cref="PayloadSchema{T}"/>.
/// </summary>
public enum FieldKind
{
    /// <summary>Exact equality on a single string (tenant, status, id, ...).</summary>
    Keyword,

    /// <summary>Equality, range and inequality comparisons on a numeric value.</summary>
    Numeric,

    /// <summary>Equality on a boolean value.</summary>
    Bool,

    /// <summary>Any-of membership over a set of strings (labels, categories, ...).</summary>
    Tag,

    /// <summary>
    /// Full-text field tokenized into the BM25 inverted index. Text fields are not queryable
    /// through the equality/range filter API; use hybrid search for them.
    /// </summary>
    Text,
}
