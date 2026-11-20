using Faro.Payload;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Payload;

public class PayloadSchemaTests
{
    [Fact]
    public void DeclaresAndExtractsEveryFieldKind()
    {
        PayloadSchema<Doc> schema = TestFixtures.Schema();
        var doc = new Doc
        {
            Tenant = "acme",
            Year = 2022,
            Active = true,
            Tags = ["red", "blue"],
            Text = "hello world",
        };

        Assert.Equal(5, schema.FieldNames.Count);
        Assert.Equal(FieldKind.Keyword, schema.KindOf("tenant"));
        Assert.Equal(FieldKind.Numeric, schema.KindOf("year"));
        Assert.Equal(FieldKind.Bool, schema.KindOf("active"));
        Assert.Equal(FieldKind.Tag, schema.KindOf("tags"));
        Assert.Equal(FieldKind.Text, schema.KindOf("text"));

        Assert.Equal("acme", schema.Extract("tenant", doc).AsString());
        Assert.Equal(2022, schema.Extract("year", doc).AsNumber());
        Assert.True(schema.Extract("active", doc).AsBool());
        Assert.Equal(new[] { "red", "blue" }, schema.Extract("tags", doc).AsTags());
        Assert.Equal("hello world", schema.Extract("text", doc).AsString());
    }

    [Fact]
    public void UnknownField_IsRejectedByKindLookupAndExtraction()
    {
        PayloadSchema<Doc> schema = TestFixtures.Schema();

        Assert.False(schema.TryGetKind("missing", out _));
        Assert.Throws<ArgumentException>(() => schema.KindOf("missing"));
        Assert.Throws<ArgumentException>(() => schema.Extract("missing", new Doc()));
    }

    [Fact]
    public void DuplicateField_Throws()
    {
        var schema = new PayloadSchema<Doc>().Keyword("tenant", d => d.Tenant);

        Assert.Throws<ArgumentException>(() => schema.Keyword("tenant", d => d.Tenant));
    }

    [Fact]
    public void EmptyOrWhitespaceName_Throws()
    {
        var schema = new PayloadSchema<Doc>();

        Assert.Throws<ArgumentException>(() => schema.Keyword("", d => d.Tenant));
        Assert.Throws<ArgumentException>(() => schema.Numeric("  ", d => d.Year));
    }

    [Fact]
    public void SecondFullTextField_Throws()
    {
        var schema = new PayloadSchema<Doc>()
            .FullText("text", d => d.Text);

        Assert.Throws<InvalidOperationException>(() => schema.FullText("other", d => d.Tenant));
    }

    [Fact]
    public void Schema_IsImmutableAfterUse()
    {
        PayloadSchema<Doc> schema = TestFixtures.Schema();
        int countBefore = schema.FieldNames.Count;

        _ = schema.Extract("year", new Doc { Year = 7 });

        Assert.Equal(countBefore, schema.FieldNames.Count);
    }
}
