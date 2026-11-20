using System.Collections.ObjectModel;

namespace Faro.Payload;

/// <summary>
/// A type-tagged payload field value used by both the payload indexes and filter evaluation.
/// </summary>
/// <remarks>
/// One struct carries all supported shapes so a schema extractor can return values without
/// boxing and a filter can ask for the right shape at evaluation time. Accessors other than the
/// one matching <see cref="Kind"/> return a neutral value (empty string, zero, false, empty
/// collection), which makes an absent or mismatched field fail the filter rather than throw.
/// </remarks>
public readonly struct FieldValue
{
    private static readonly IReadOnlyList<string> EmptyTags = Array.Empty<string>();

    private readonly string? _string;
    private readonly ReadOnlyCollection<string>? _tags;

    private FieldValue(
        FieldKind kind,
        string? text,
        ReadOnlyCollection<string>? tags,
        double number,
        bool boolean)
    {
        Kind = kind;
        _string = text;
        _tags = tags;
        Number = number;
        Bool = boolean;
    }

    /// <summary>Shape of this value.</summary>
    public FieldKind Kind { get; }

    /// <summary>Numeric payload, valid when <see cref="Kind"/> is <see cref="FieldKind.Numeric"/>.</summary>
    public double Number { get; }

    /// <summary>Boolean payload, valid when <see cref="Kind"/> is <see cref="FieldKind.Bool"/>.</summary>
    public bool Bool { get; }

    /// <summary>Gets <see langword="true"/> when no value has been assigned (the default struct).</summary>
    public bool IsMissing => Kind == default && _string is null && _tags is null && Number == 0 && !Bool;

    /// <summary>Creates a keyword or text value.</summary>
    /// <param name="value">The string, or <see langword="null"/> for a missing value.</param>
    public static FieldValue String(string? value) =>
        new(FieldKind.Keyword, value, null, 0, false);

    /// <summary>Creates a numeric value.</summary>
    /// <param name="value">The numeric value.</param>
    public static FieldValue NumberValue(double value) =>
        new(FieldKind.Numeric, null, null, value, false);

    /// <summary>Creates a boolean value.</summary>
    /// <param name="value">The boolean value.</param>
    public static FieldValue Boolean(bool value) =>
        new(FieldKind.Bool, null, null, 0, value);

    /// <summary>Creates a tags value. The sequence is copied into a read-only snapshot.</summary>
    /// <param name="values">The tags, or <see langword="null"/> for no tags.</param>
    public static FieldValue Tags(IEnumerable<string?>? values)
    {
        if (values is null)
        {
            return new FieldValue(FieldKind.Tag, null, null, 0, false);
        }

        var copy = new List<string>();
        foreach (string? value in values)
        {
            if (value is not null)
            {
                copy.Add(value);
            }
        }

        return new FieldValue(FieldKind.Tag, null, copy.AsReadOnly(), 0, false);
    }

    /// <summary>Gets the string payload for keyword/text values; otherwise an empty string.</summary>
    public string AsString() => Kind is FieldKind.Keyword or FieldKind.Text ? _string ?? string.Empty : string.Empty;

    /// <summary>Gets the numeric payload for numeric values; otherwise zero.</summary>
    public double AsNumber() => Kind == FieldKind.Numeric ? Number : 0;

    /// <summary>Gets the boolean payload for boolean values; otherwise <see langword="false"/>.</summary>
    public bool AsBool() => Kind == FieldKind.Bool && Bool;

    /// <summary>Gets the tag payload for tag values; otherwise an empty collection.</summary>
    public IReadOnlyList<string> AsTags() => Kind == FieldKind.Tag ? _tags ?? EmptyTags : EmptyTags;
}
