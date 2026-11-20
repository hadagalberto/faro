using Faro.Payload;
using Xunit;

namespace Faro.Tests.Payload;

public class FieldValueTests
{
    [Fact]
    public void String_KeepsValueAndKind()
    {
        FieldValue value = FieldValue.String("acme");

        Assert.Equal(FieldKind.Keyword, value.Kind);
        Assert.Equal("acme", value.AsString());
        Assert.Equal(0, value.AsNumber());
        Assert.False(value.AsBool());
        Assert.Empty(value.AsTags());
    }

    [Fact]
    public void NumberAndBool_ExposeOnlyTheirOwnShape()
    {
        FieldValue number = FieldValue.NumberValue(2023.5);
        FieldValue boolean = FieldValue.Boolean(true);

        Assert.Equal(FieldKind.Numeric, number.Kind);
        Assert.Equal(2023.5, number.AsNumber());
        Assert.Equal(string.Empty, number.AsString());
        Assert.True(boolean.AsBool());
        Assert.Equal(FieldKind.Bool, boolean.Kind);
        Assert.False(FieldValue.Boolean(false).AsBool());
    }

    [Fact]
    public void Tags_CopiesInputAndSkipsNulls()
    {
        string[] source = ["a", "b"];
        FieldValue value = FieldValue.Tags(source);

        source[0] = "mutated";
        Assert.Equal(new[] { "a", "b" }, value.AsTags());
        Assert.Equal(FieldKind.Tag, value.Kind);
    }

    [Fact]
    public void Tags_NullYieldsEmptySnapshot()
    {
        FieldValue value = FieldValue.Tags(null);

        Assert.Equal(FieldKind.Tag, value.Kind);
        Assert.Empty(value.AsTags());
    }
}
