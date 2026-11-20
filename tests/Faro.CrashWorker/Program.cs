using System.Text.Json.Serialization;
using Faro;
using Faro.Payload;

// Child process used by WalDurabilityTests. It writes one batch and dies with FailFast so no
// Dispose/finalizer can flush on the way out: only data already fsynced by the library survives.
if (args.Length < 3)
{
    throw new ArgumentException("expected: <db path> <durability> <record count>");
}

string path = args[0];
var durability = (Durability)int.Parse(args[1]);
int count = int.Parse(args[2]);

using var db = FaroDb.Open(path, new FaroOptions { Durability = durability });
FaroDbCollection<WorkerDoc> docs = db.GetOrCreateCollection("docs", new CollectionOptions<WorkerDoc>
{
    Dimensions = 16,
    Distance = Distance.DotProduct,
    Index = new HnswOptions { M = 8, EfConstruction = 64, EfSearch = 32, Seed = 0x5EED },
    Serializer = new SourceGenPayloadSerializer<WorkerDoc>(WorkerContext.Default.WorkerDoc),
    Schema = new PayloadSchema<WorkerDoc>()
        .Keyword("tenant", d => d.Tenant)
        .Numeric("year", d => d.Year)
        .Bool("active", d => d.Active)
        .Tags("tags", d => d.Tags)
        .FullText("text", d => d.Text),
    DefaultEfSearch = 32,
});

var items = new List<(int Key, float[] Vector, WorkerDoc Payload)>(count);
for (int i = 0; i < count; i++)
{
    var vector = new float[16];
    vector[i % 16] = 1f;
    items.Add((i, vector, new WorkerDoc { Tenant = "acme", Year = 2020 + i, Text = "gato" }));
}

docs.UpsertBatch(items);

Environment.FailFast("faro-crash-probe");

internal sealed class WorkerDoc
{
    public string Tenant { get; set; } = string.Empty;
    public int Year { get; set; }
    public bool Active { get; set; }
    public string[] Tags { get; set; } = [];
    public string Text { get; set; } = string.Empty;
}

[JsonSerializable(typeof(WorkerDoc))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class WorkerContext : JsonSerializerContext
{
}
