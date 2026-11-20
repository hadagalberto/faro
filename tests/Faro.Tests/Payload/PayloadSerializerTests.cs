using System.Text;
using System.Text.Json;
using Faro.Payload;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Payload;

public class PayloadSerializerTests
{
    [Fact]
    public void SourceGenSerializer_RoundTripsKnownPayload()
    {
        IPayloadSerializer<Doc> serializer = TestFixtures.Serializer();
        var original = new Doc
        {
            Tenant = "acme",
            Year = 2024,
            Active = true,
            Tags = ["one", "two"],
            Text = "ação na árvore",
        };

        byte[] bytes = serializer.Serialize(original);
        Doc? copy = serializer.Deserialize(bytes);

        Assert.NotNull(copy);
        Assert.Equal(original.Tenant, copy.Tenant);
        Assert.Equal(original.Year, copy.Year);
        Assert.Equal(original.Active, copy.Active);
        Assert.Equal(original.Tags, copy.Tags);
        Assert.Equal(original.Text, copy.Text);
    }

    [Fact]
    public void SourceGenSerializer_UsesCamelCasePropertyNames()
    {
        var serializer = TestFixtures.Serializer();
        string json = Encoding.UTF8.GetString(serializer.Serialize(new Doc { Tenant = "acme" }));

        Assert.Contains("\"tenant\":\"acme\"", json);
        Assert.Contains("\"year\":0", json);
        Assert.DoesNotContain("\"Tenant\"", json);
    }

    [Fact]
    public void SourceGenSerializer_KnownInvalidJson_ThrowsJsonException()
    {
        var serializer = TestFixtures.Serializer();

        Assert.Throws<JsonException>(() => serializer.Deserialize(Encoding.UTF8.GetBytes("{ not json")));
    }

    [Fact]
    public void SourceGenSerializer_NullArguments_Throw()
    {
        var serializer = TestFixtures.Serializer();

        Assert.Throws<ArgumentNullException>(() => serializer.Serialize(null!));
        Assert.Throws<ArgumentNullException>(() => new SourceGenPayloadSerializer<Doc>(null!));
    }

    [Fact]
    public void SourceGenSerializer_JsonNull_DeserializesToNull()
    {
        var serializer = TestFixtures.Serializer();

        Assert.Null(serializer.Deserialize(Encoding.UTF8.GetBytes("null")));
    }
}
