using System.Text.Json.Serialization;

namespace Faro.Tests.Shared;

/// <summary>Payload used by the milestone-4 unit tests.</summary>
public sealed class Doc
{
    public string Tenant { get; set; } = string.Empty;

    public int Year { get; set; }

    public bool Active { get; set; }

    public string[] Tags { get; set; } = [];

    public string Text { get; set; } = string.Empty;
}

/// <summary>Source-generated metadata for <see cref="Doc"/>; no reflection at runtime.</summary>
[JsonSerializable(typeof(Doc))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class TestJsonContext : JsonSerializerContext
{
}

internal static class TestFixtures
{
    internal static global::Faro.Payload.PayloadSchema<Doc> Schema() =>
        new global::Faro.Payload.PayloadSchema<Doc>()
            .Keyword("tenant", d => d.Tenant)
            .Numeric("year", d => d.Year)
            .Bool("active", d => d.Active)
            .Tags("tags", d => d.Tags)
            .FullText("text", d => d.Text);

    internal static global::Faro.Payload.SourceGenPayloadSerializer<Doc> Serializer() =>
        new(TestJsonContext.Default.Doc);
}
