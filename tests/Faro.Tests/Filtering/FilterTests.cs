using Faro.Filtering;
using Faro.Payload;
using Xunit;

namespace Faro.Tests.Filtering;

public class FilterTests
{
    private static FilterIndexView View()
    {
        var view = new FilterIndexView(new HashSet<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

        var tenants = new TermIndex();
        tenants.Add(1, FieldValue.String("acme"));
        tenants.Add(2, FieldValue.String("acme"));
        tenants.Add(3, FieldValue.String("globex"));
        view.Add("tenant", tenants);

        var years = new NumericIndex();
        years.Add(1, 2019);
        years.Add(2, 2021);
        years.Add(3, 2023);
        years.Add(4, 2025);
        view.Add("year", years);

        var active = new BoolIndex();
        active.Add(1, true);
        active.Add(2, false);
        active.Add(3, true);
        view.Add("active", active);

        var tags = new TermIndex();
        tags.Add(1, FieldValue.Tags(["red", "blue"]));
        tags.Add(2, FieldValue.Tags(["green"]));
        tags.Add(4, FieldValue.Tags(["red"]));
        view.Add("tags", tags);

        return view;
    }

    private static int[] Ids(HashSet<int> set) => set.OrderBy(id => id).ToArray();

    [Fact]
    public void Eq_Keyword_SelectsOnlyMatchingIds()
    {
        Assert.Equal([1, 2], Ids(Filter.Eq("tenant", "acme").Evaluate(View())));
        Assert.Empty(Filter.Eq("tenant", "missing").Evaluate(View()));
    }

    [Fact]
    public void Eq_BoolAndNumber_SelectOnlyTheirFieldKinds()
    {
        Assert.Equal([1, 3], Ids(Filter.Eq("active", true).Evaluate(View())));
        Assert.Equal([2], Ids(Filter.Eq("active", false).Evaluate(View())));
        Assert.Equal([2], Ids(Filter.Eq("year", 2021).Evaluate(View())));
    }

    [Fact]
    public void Range_IsInclusiveAndComparisonsAreExclusiveWhereExpected()
    {
        FilterIndexView view = View();

        Assert.Equal([2, 3], Ids(Filter.Range("year", 2020, 2024).Evaluate(view)));
        Assert.Equal([1, 2, 3], Ids(Filter.Le("year", 2023).Evaluate(view)));
        Assert.Equal([3, 4], Ids(Filter.Ge("year", 2023).Evaluate(view)));
        Assert.Equal([3, 4], Ids(Filter.Gt("year", 2022).Evaluate(view)));
        Assert.Equal([1, 2], Ids(Filter.Lt("year", 2023).Evaluate(view)));
    }

    [Fact]
    public void In_UnionsTagValues()
    {
        Assert.Equal([1, 2, 4], Ids(Filter.In("tags", "red", "green").Evaluate(View())));
    }

    [Fact]
    public void ComposableAndOrNot_UseSetSemantics()
    {
        FilterIndexView view = View();
        Filter acme = Filter.Eq("tenant", "acme");
        Filter recent = Filter.Range("year", 2020, 2024);

        Assert.Equal([2], Ids((acme & recent).Evaluate(view)));
        Assert.Equal([1, 2, 3], Ids((acme | recent).Evaluate(view)));
        Assert.Equal([3, 4, 5, 6, 7, 8, 9, 10], Ids((!acme).Evaluate(view)));
        Assert.Equal([1], Ids((acme & Filter.Eq("active", true)).Evaluate(view)));
    }

    [Fact]
    public void Operators_DoNotMutateOperandsAndCanBeReused()
    {
        Filter a = Filter.Eq("tenant", "acme");
        Filter b = Filter.Eq("active", true);
        Filter composed = a & b;

        Assert.Equal([1, 2], Ids(a.Evaluate(View())));
        Assert.Equal([1, 3], Ids(b.Evaluate(View())));
        Assert.Equal([1], Ids(composed.Evaluate(View())));
        Assert.Equal([1], Ids(composed.Evaluate(View())));
    }

    [Fact]
    public void All_MatchesEverythingAndIsNeutralForAnd()
    {
        Assert.Equal(Enumerable.Range(1, 10).ToArray(), Ids(Filter.All.Evaluate(View())));
        Filter eq = Filter.Eq("tenant", "acme");
        Assert.Equal([1, 2], Ids((Filter.All & eq).Evaluate(View())));
        Assert.Same(eq, Filter.And(Filter.All, eq));
        Assert.Same(eq, Filter.And(eq, Filter.All));
    }

    [Fact]
    public void Matches_EvaluatesEveryLeafAgainstOnePayload()
    {
        var values = new Lookup
        {
            ["tenant"] = FieldValue.String("acme"),
            ["year"] = FieldValue.NumberValue(2021),
            ["active"] = FieldValue.Boolean(true),
            ["tags"] = FieldValue.Tags(["red", "blue"]),
        };
        FilterIndexView view = View();

        Assert.True(Filter.Eq("tenant", "acme").Matches(values));
        Assert.True(Filter.Range("year", 2020, 2022).Matches(values));
        Assert.True(Filter.Eq("active", true).Matches(values));
        Assert.True(Filter.In("tags", "green", "red").Matches(values));
        Assert.False(Filter.In("tags", "green").Matches(values));
        Assert.True((Filter.Eq("tenant", "acme") & Filter.Gt("year", 2020)).Matches(values));
        Assert.False((Filter.Eq("tenant", "acme") & Filter.Gt("year", 2023)).Matches(values));
        Assert.True((!Filter.Eq("tenant", "globex")).Matches(values));

        // Index-based evaluation and per-record evaluation agree on the same filter.
        Assert.Equal(new[] { 1, 2 }, Ids(Filter.Eq("tenant", "acme").Evaluate(view)));
    }

    [Fact]
    public void Matches_MissingFieldFailsClosed()
    {
        var values = new Lookup();

        Assert.False(Filter.Eq("tenant", "acme").Matches(values));
        Assert.False(Filter.Range("year", 2020, 2022).Matches(values));
        Assert.False(Filter.Eq("active", true).Matches(values));
    }

    [Fact]
    public void UnknownField_Throws()
    {
        Assert.Throws<ArgumentException>(() => Filter.Eq("unknown", "x").Evaluate(View()));
        Assert.Throws<ArgumentException>(() => Filter.Range("unknown", 1, 2).Evaluate(View()));
    }

    [Fact]
    public void WrongFieldKind_Throws()
    {
        Assert.Throws<ArgumentException>(() => Filter.Eq("year", "x").Evaluate(View()));
        Assert.Throws<ArgumentException>(() => Filter.Range("tenant", 1, 2).Evaluate(View()));
    }

    [Fact]
    public void InvalidArguments_AreRejected()
    {
        Assert.Throws<ArgumentException>(() => Filter.Eq("", "x"));
        Assert.Throws<ArgumentNullException>(() => Filter.Eq("tenant", null!));
        Assert.Throws<ArgumentException>(() => Filter.In("tags"));
        Assert.Throws<ArgumentException>(() => Filter.In("tags", "a", null!));
        Assert.Throws<ArgumentException>(() => Filter.Range("year", 5, 1));
        Assert.Throws<ArgumentException>(() => Filter.Eq("year", double.NaN));
        Assert.Throws<ArgumentNullException>(() => Filter.Not(null!));
        Assert.Throws<ArgumentNullException>(() => Filter.And(null!, Filter.All));
    }

    [Fact]
    public void DoubleNegation_IsSimplified()
    {
        Filter inner = Filter.Eq("tenant", "acme");

        Assert.Same(inner, Filter.Not(Filter.Not(inner)));
    }

    [Fact]
    public void ToString_ShapesComposedFilters()
    {
        string text = (Filter.Eq("tenant", "acme") & Filter.Range("year", 2020, 2024)).ToString();

        Assert.Contains("tenant", text);
        Assert.Contains("year", text);
        Assert.Contains("&", text);
    }

    private sealed class Lookup : Dictionary<string, FieldValue>, IFieldValueLookup
    {
        public bool TryGet(string field, out FieldValue value) => TryGetValue(field, out value);
    }
}
