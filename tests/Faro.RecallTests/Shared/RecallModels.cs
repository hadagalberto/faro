using System.Text.Json.Serialization;
using Faro.Payload;

namespace Faro.RecallTests.Shared;

/// <summary>Minimal payload for filtered recall tests.</summary>
public sealed class RecallDoc
{
    public string Tenant { get; set; } = string.Empty;

    public int Year { get; set; }
}

[JsonSerializable(typeof(RecallDoc))]
internal sealed partial class RecallJsonContext : JsonSerializerContext
{
}

internal static class RecallFixtures
{
    internal static PayloadSchema<RecallDoc> Schema() =>
        new PayloadSchema<RecallDoc>()
            .Keyword("tenant", d => d.Tenant)
            .Numeric("year", d => d.Year);

    internal static SourceGenPayloadSerializer<RecallDoc> Serializer() =>
        new(RecallJsonContext.Default.RecallDoc);
}
