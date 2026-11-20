using Faro.Payload;

namespace Faro.Filtering;

/// <summary>
/// Read-only view over one stored payload, used to evaluate a filter during graph traversal.
/// </summary>
internal interface IFieldValueLookup
{
    bool TryGet(string field, out FieldValue value);
}

/// <summary>
/// An immutable, composable predicate over indexed payload fields.
/// </summary>
/// <remarks>
/// <para>
/// Leaves are equality (<see cref="Eq(string, string)"/>), numeric range
/// (<see cref="Range(string, double, double)"/>), any-of membership (<see cref="In"/>) and
/// comparisons. <see cref="And"/>/<see cref="Or"/> are expressed with <c>&amp;</c> and
/// <c>|</c>, negation with <c>!</c>. Operators never mutate their operands, so a filter can be
/// built once and reused across searches and collections.
/// </para>
/// <para>
/// A filter is evaluated in two ways: <c>Evaluate</c> over the payload indexes returns the exact
/// candidate id set and its selectivity (used by the planner), while <c>Matches</c> checks one
/// stored payload when the planner chooses filtered HNSW traversal.
/// </para>
/// </remarks>
public abstract class Filter
{
    /// <summary>A filter that matches every stored vector.</summary>
    public static Filter All { get; } = new AllFilter();

    /// <summary>Creates an equality filter on a keyword or tag field.</summary>
    /// <param name="field">The declared field name.</param>
    /// <param name="value">The value to match.</param>
    /// <returns>The filter.</returns>
    public static Filter Eq(string field, string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(field);
        ArgumentNullException.ThrowIfNull(value);
        return new StringEqFilter(field, value);
    }

    /// <summary>Creates an equality filter on a boolean field.</summary>
    /// <param name="field">The declared field name.</param>
    /// <param name="value">The value to match.</param>
    /// <returns>The filter.</returns>
    public static Filter Eq(string field, bool value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(field);
        return new BoolEqFilter(field, value);
    }

    /// <summary>Creates an equality filter on a numeric field.</summary>
    /// <param name="field">The declared field name.</param>
    /// <param name="value">The value to match.</param>
    /// <returns>The filter.</returns>
    public static Filter Eq(string field, double value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(field);
        if (double.IsNaN(value))
        {
            throw new ArgumentException("NaN cannot equal any stored value", nameof(value));
        }

        return new NumberEqFilter(field, value);
    }

    /// <summary>Creates an equality filter on a numeric field.</summary>
    /// <param name="field">The declared field name.</param>
    /// <param name="value">The value to match.</param>
    /// <returns>The filter.</returns>
    public static Filter Eq(string field, long value) => Eq(field, (double)value);

    /// <summary>Creates an inclusive numeric range filter.</summary>
    /// <param name="field">The declared numeric field name.</param>
    /// <param name="min">Lower bound, inclusive.</param>
    /// <param name="max">Upper bound, inclusive.</param>
    /// <returns>The filter.</returns>
    /// <exception cref="ArgumentException"><paramref name="min"/> is greater than <paramref name="max"/>, or a bound is NaN.</exception>
    public static Filter Range(string field, double min, double max) =>
        Range(field, min, max, includeMin: true, includeMax: true);

    /// <summary>Creates an inclusive numeric range filter from integer bounds.</summary>
    /// <param name="field">The declared numeric field name.</param>
    /// <param name="min">Lower bound, inclusive.</param>
    /// <param name="max">Upper bound, inclusive.</param>
    /// <returns>The filter.</returns>
    public static Filter Range(string field, long min, long max) => Range(field, (double)min, (double)max);

    /// <summary>Creates a filter matching any of the supplied keyword or tag values.</summary>
    /// <param name="field">The declared field name.</param>
    /// <param name="values">The accepted values. The array is copied.</param>
    /// <returns>The filter.</returns>
    /// <exception cref="ArgumentException"><paramref name="values"/> is empty.</exception>
    public static Filter In(string field, params string[] values)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(field);
        ArgumentNullException.ThrowIfNull(values);
        if (values.Length == 0)
        {
            throw new ArgumentException("In requires at least one value", nameof(values));
        }

        var copy = new string[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            copy[i] = values[i] ?? throw new ArgumentException("In values must not be null", nameof(values));
        }

        return new InFilter(field, copy);
    }

    /// <summary>Creates a strict greater-than filter on a numeric field.</summary>
    /// <param name="field">The declared numeric field name.</param>
    /// <param name="value">Exclusive lower bound.</param>
    /// <returns>The filter.</returns>
    public static Filter Gt(string field, double value) =>
        Range(field, value, double.PositiveInfinity, includeMin: false, includeMax: true);

    /// <summary>Creates a greater-or-equal filter on a numeric field.</summary>
    /// <param name="field">The declared numeric field name.</param>
    /// <param name="value">Inclusive lower bound.</param>
    /// <returns>The filter.</returns>
    public static Filter Ge(string field, double value) =>
        Range(field, value, double.PositiveInfinity, includeMin: true, includeMax: true);

    /// <summary>Creates a strict less-than filter on a numeric field.</summary>
    /// <param name="field">The declared numeric field name.</param>
    /// <param name="value">Exclusive upper bound.</param>
    /// <returns>The filter.</returns>
    public static Filter Lt(string field, double value) =>
        Range(field, double.NegativeInfinity, value, includeMin: true, includeMax: false);

    /// <summary>Creates a less-or-equal filter on a numeric field.</summary>
    /// <param name="field">The declared numeric field name.</param>
    /// <param name="value">Inclusive upper bound.</param>
    /// <returns>The filter.</returns>
    public static Filter Le(string field, double value) =>
        Range(field, double.NegativeInfinity, value, includeMin: true, includeMax: true);

    /// <summary>Negates a filter.</summary>
    /// <param name="filter">The filter to negate.</param>
    /// <returns>The negated filter.</returns>
    public static Filter Not(Filter filter)
    {
        ArgumentNullException.ThrowIfNull(filter);
        return filter is NotFilter negation ? negation.Inner : new NotFilter(filter);
    }

    /// <summary>Combines two filters with AND.</summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns>The composed filter.</returns>
    public static Filter And(Filter left, Filter right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        if (ReferenceEquals(left, All))
        {
            return right;
        }

        return ReferenceEquals(right, All) ? left : new AndFilter(left, right);
    }

    /// <summary>Combines two filters with OR.</summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns>The composed filter.</returns>
    public static Filter Or(Filter left, Filter right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        if (ReferenceEquals(left, All) || ReferenceEquals(right, All))
        {
            return All;
        }

        return new OrFilter(left, right);
    }

    /// <summary>Composes two filters with AND.</summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns>The composed filter.</returns>
    public static Filter operator &(Filter left, Filter right) => And(left, right);

    /// <summary>Composes two filters with OR.</summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns>The composed filter.</returns>
    public static Filter operator |(Filter left, Filter right) => Or(left, right);

    /// <summary>Negates a filter.</summary>
    /// <param name="filter">The filter to negate.</param>
    /// <returns>The negated filter.</returns>
    public static Filter operator !(Filter filter) => Not(filter);

    /// <inheritdoc />
    public override string ToString() => Describe();

    internal bool IsAll => this is AllFilter;

    internal abstract HashSet<int> Evaluate(FilterIndexView view);

    internal abstract bool Matches(IFieldValueLookup values);

    internal abstract void CollectFields(HashSet<string> fields);

    internal abstract string Describe();

    private static Filter Range(string field, double min, double max, bool includeMin, bool includeMax)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(field);
        if (double.IsNaN(min) || double.IsNaN(max) || min > max)
        {
            throw new ArgumentException("range bounds must be ordered and not NaN", nameof(min));
        }

        return new RangeFilter(field, min, max, includeMin, includeMax);
    }

    private sealed class AllFilter : Filter
    {
        internal override HashSet<int> Evaluate(FilterIndexView view) => new(view.AllIds);

        internal override bool Matches(IFieldValueLookup values) => true;

        internal override void CollectFields(HashSet<string> fields)
        {
        }

        internal override string Describe() => "all";
    }

    private sealed class StringEqFilter(string field, string value) : Filter
    {
        internal override HashSet<int> Evaluate(FilterIndexView view) => view.Term(field).Get(value);

        internal override bool Matches(IFieldValueLookup values)
        {
            if (!values.TryGet(field, out FieldValue stored))
            {
                return false;
            }

            return stored.Kind switch
            {
                FieldKind.Keyword or FieldKind.Text => string.Equals(stored.AsString(), value, StringComparison.Ordinal),
                FieldKind.Tag => ContainsTag(stored, value),
                _ => false,
            };
        }

        internal override void CollectFields(HashSet<string> fields) => fields.Add(field);

        internal override string Describe() => $"{field} = \"{value}\"";

        private static bool ContainsTag(FieldValue stored, string value)
        {
            IReadOnlyList<string> tags = stored.AsTags();
            for (int i = 0; i < tags.Count; i++)
            {
                if (string.Equals(tags[i], value, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }
    }

    private sealed class BoolEqFilter(string field, bool value) : Filter
    {
        internal override HashSet<int> Evaluate(FilterIndexView view) => view.Bool(field).Get(value);

        internal override bool Matches(IFieldValueLookup values) =>
            values.TryGet(field, out FieldValue stored) && stored.Kind == FieldKind.Bool && stored.AsBool() == value;

        internal override void CollectFields(HashSet<string> fields) => fields.Add(field);

        internal override string Describe() => $"{field} = {(value ? "true" : "false")}";
    }

    private sealed class NumberEqFilter(string field, double value) : Filter
    {
        internal override HashSet<int> Evaluate(FilterIndexView view) => view.Numeric(field).Get(value);

        internal override bool Matches(IFieldValueLookup values) =>
            values.TryGet(field, out FieldValue stored) && stored.Kind == FieldKind.Numeric && stored.Number == value;

        internal override void CollectFields(HashSet<string> fields) => fields.Add(field);

        internal override string Describe() => $"{field} = {value}";
    }

    private sealed class RangeFilter(
        string field,
        double min,
        double max,
        bool includeMin,
        bool includeMax) : Filter
    {
        internal override HashSet<int> Evaluate(FilterIndexView view) =>
            view.Numeric(field).Range(min, max, includeMin, includeMax);

        internal override bool Matches(IFieldValueLookup values)
        {
            if (!values.TryGet(field, out FieldValue stored) || stored.Kind != FieldKind.Numeric)
            {
                return false;
            }

            double value = stored.Number;
            bool aboveMin = includeMin ? value >= min : value > min;
            bool belowMax = includeMax ? value <= max : value < max;
            return aboveMin && belowMax;
        }

        internal override void CollectFields(HashSet<string> fields) => fields.Add(field);

        internal override string Describe() =>
            $"{field} {(includeMin ? ">=" : ">")} {min} {(includeMax ? "<=" : "<")} {max}";
    }

    private sealed class InFilter(string field, string[] values) : Filter
    {
        internal override HashSet<int> Evaluate(FilterIndexView view) => view.Term(field).GetAny(values);

        internal override bool Matches(IFieldValueLookup values)
        {
            if (!values.TryGet(field, out FieldValue stored))
            {
                return false;
            }

            if (stored.Kind is FieldKind.Keyword or FieldKind.Text)
            {
                return Contains(stored.AsString());
            }

            if (stored.Kind == FieldKind.Tag)
            {
                IReadOnlyList<string> tags = stored.AsTags();
                for (int i = 0; i < tags.Count; i++)
                {
                    if (Contains(tags[i]))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        internal override void CollectFields(HashSet<string> fields) => fields.Add(field);

        internal override string Describe() => $"{field} in [{string.Join(", ", values)}]";

        private bool Contains(string candidate)
        {
            for (int i = 0; i < values.Length; i++)
            {
                if (string.Equals(values[i], candidate, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }
    }

    private sealed class AndFilter(Filter left, Filter right) : Filter
    {
        internal override HashSet<int> Evaluate(FilterIndexView view)
        {
            HashSet<int> a = left.Evaluate(view);
            HashSet<int> b = right.Evaluate(view);
            if (a.Count > b.Count)
            {
                (a, b) = (b, a);
            }

            a.IntersectWith(b);
            return a;
        }

        internal override bool Matches(IFieldValueLookup values) => left.Matches(values) && right.Matches(values);

        internal override void CollectFields(HashSet<string> fields)
        {
            left.CollectFields(fields);
            right.CollectFields(fields);
        }

        internal override string Describe() => $"({left} & {right})";
    }

    private sealed class OrFilter(Filter left, Filter right) : Filter
    {
        internal override HashSet<int> Evaluate(FilterIndexView view)
        {
            HashSet<int> result = left.Evaluate(view);
            result.UnionWith(right.Evaluate(view));
            return result;
        }

        internal override bool Matches(IFieldValueLookup values) => left.Matches(values) || right.Matches(values);

        internal override void CollectFields(HashSet<string> fields)
        {
            left.CollectFields(fields);
            right.CollectFields(fields);
        }

        internal override string Describe() => $"({left} | {right})";
    }

    private sealed class NotFilter(Filter inner) : Filter
    {
        internal Filter Inner => inner;

        internal override HashSet<int> Evaluate(FilterIndexView view)
        {
            var result = new HashSet<int>(view.AllIds);
            result.ExceptWith(inner.Evaluate(view));
            return result;
        }

        internal override bool Matches(IFieldValueLookup values) => !inner.Matches(values);

        internal override void CollectFields(HashSet<string> fields) => inner.CollectFields(fields);

        internal override string Describe() => $"!({inner})";
    }
}
