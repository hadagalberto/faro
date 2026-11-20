# Faro

**Faro** is an embedded, zero-server vector database for .NET — "SQLite for embeddings". One NuGet package, one file on disk, SIMD distance kernels, exact and approximate indexes, and a path into Semantic Kernel / Microsoft.Extensions.AI RAG pipelines. MIT licensed, targets `net8.0` and `net10.0`, and is built to be Native AOT and trimming compatible.

> **Status — Milestone 5:** on top of exact/HNSW indexes, quantization with rescoring, typed payloads, filters and BM25 hybrid search, `FaroDb` now provides a versioned single-file format, a CRC32C write-ahead log with configurable durability, torn-tail crash recovery, atomic checkpoints, memory-mapped vector reads, in-memory mode, generic keys and sync/async batch APIs. The `Microsoft.Extensions.VectorData` connector is still planned; see [docs/architecture.md](docs/architecture.md).

## Why Faro

Most .NET HNSW packages hand you a bare index. Faro aims at the complete database around it: measured recall, quantization with rescoring, filters that stay accurate under high selectivity, hybrid vector + BM25 search, and crash-safe single-file persistence. The exact `FlatIndex` in this release is the ground-truth engine every later approximate index is validated against.

| Feature | Faro (today) | Plain HNSW library (typical) |
| --- | --- | --- |
| Exact Flat index with SIMD kernels and parallel search | ✅ | — |
| SIMD + scalar reference kernels with equivalence tests | ✅ | sometimes |
| Pooled heaps / visited sets (no per-search allocations) | ✅ | — |
| HNSW with deterministic builds, concurrent inserts and neighbour repair | ✅ | ✅ |
| Scalar int8 / binary quantization with quantile clipping and rescoring | ✅ | — |
| Payload filters with selectivity-aware planning | ✅ | — |
| BM25 full-text + RRF hybrid search | ✅ | — |
| Single-file persistence with CRC32C WAL + crash recovery | ✅ | — |
| Microsoft.Extensions.VectorData connector | planned | — |

## Quick start

```csharp
using Faro;

var index = new FlatIndex(new FlatOptions
{
    Dimensions = 768,
    Distance = Distance.Cosine, // vectors are L2-normalized on add by default
    ParallelThreshold = 4096,   // collections this large search with Parallel.For
});

int id = index.Add(embedding);
SearchResult[] hits = index.Search(queryEmbedding, k: 10);
foreach (var hit in hits)
{
    Console.WriteLine($"{hit.Id}: {hit.Score:F4}");
}
```

For a zero-allocation steady state, call the span overload with a caller-owned buffer:

```csharp
var buffer = new SearchResult[10];
int written = index.Search(queryEmbedding, k: 10, buffer);
```

Results are ordered best-first — ascending score for Cosine / Euclidean / Hamming, descending for DotProduct — with exact ties broken by ascending id. The sequential search path allocates nothing except the returned array of the allocating overload.

## Approximate search (HNSW)

```csharp
var index = new HnswIndex(new HnswOptions
{
    Dimensions = 768,
    Distance = Distance.Cosine,
    M = 16,               // layer-0 degree bound is 2 * M
    EfConstruction = 200,
    EfSearch = 64,
    Seed = 12345,         // omit for randomized levels
});

int id = index.Add(embedding);
SearchResult[] hits = index.Search(queryEmbedding, k: 10);          // uses EfSearch
SearchResult[] wider = index.Search(queryEmbedding, k: 10, efSearch: 128);
index.Delete(id);         // tombstone + neighbour repair
index.Compact();          // rebuild without deleted nodes, ids unchanged
```

`ExtendCandidates` (default on) and `KeepPrunedConnections` control Algorithm 4; `Mc` relaxes its diversity test. Inserts may run in parallel with each other and with searches. The span `Search(query, k, buffer)` overload is allocation-free in steady state. Recall gates live in `tests/Faro.RecallTests`.

## Quantization

Both indexes accept an optional `Quantization` option. Searches score compact codes first, keep `k * rescore` candidates, then rescore them against the stored full-precision vectors before returning the best `k`. The returned score is always the full-precision one.

```csharp
// Train on a representative sample (recommended for reproducible ranges).
var scalar = ScalarQuantizer.Train(sample, Dimensions, Distance.Cosine, quantile: 0.995);
var index = new HnswIndex(new HnswOptions
{
    Dimensions = 768,
    Quantization = Quantization.Int8(scalar, rescore: 4),
});

// Or train lazily from the first vectors added to the index:
var flat8 = new FlatIndex(new FlatOptions
{
    Dimensions = 768,
    Quantization = Quantization.Int8(rescore: 4, quantile: 0.995),
});

// Binary sign codes (1 bit/dimension) with a wider rescore window:
var binary = new FlatIndex(new FlatOptions
{
    Dimensions = 768,
    Quantization = Quantization.Binary(rescore: 8),
});
```

`ScalarQuantizer` trains per-dimension `min`/`max` (or quantile-clipped bounds) and encodes one signed byte per dimension; int8 distances use `Vector512/256/128` integer kernels. `BinaryQuantizer` packs signs 64 per word and uses popcount Hamming. Graph construction always uses full precision; quantization only steers search traversal. Measured recall is in `tests/Faro.RecallTests/QuantizationRecallTests.cs`.

## Payloads, filtering and hybrid search

`FaroCollection<T>` binds a vector index to typed payloads. Serialization uses a caller-supplied source-generated `JsonTypeInfo<T>` (no reflection, AOT-safe); field selectors are delegates, so no member names are discovered at runtime.

```csharp
using Faro;
using Faro.Filtering;
using Faro.Hybrid;
using Faro.Payload;
using System.Text.Json.Serialization;

[JsonSerializable(typeof(Doc))]
internal sealed partial class DocContext : JsonSerializerContext { }

var schema = new PayloadSchema<Doc>()
    .Keyword("tenant", d => d.Tenant)
    .Numeric("year", d => d.Year)
    .Bool("active", d => d.Active)
    .Tags("tags", d => d.Tags)
    .FullText("text", d => d.Text); // feeds the BM25 index

var docs = new FaroCollection<Doc>(
    schema,
    new SourceGenPayloadSerializer<Doc>(DocContext.Default.Doc),
    new HnswOptions { Dimensions = 768, Distance = Distance.Cosine });

int id = docs.Add(embedding, doc);
docs.TryGetPayload(id, out Doc? stored);

Filter filter = (Filter.Eq("tenant", "acme") & Filter.Range("year", 2020, 2024))
                | Filter.In("tags", "ai", "rag");
SearchResult[] hits = docs.Search(queryEmbedding, k: 10, filter: filter);
SearchDecision decision = docs.LastDecision; // ExactCandidates or FilteredTraversal

HybridResult[] hybrid = docs.HybridSearch(
    queryEmbedding, "ação", k: 10, fusion: Fusion.Rrf(60));
```

Filters are immutable and composable with `&`, `|` and `!`. The planner estimates selectivity from the payload indexes: a filter matching few vectors is scored exactly over its candidate ids; a broad filter traverses the HNSW graph with the predicate applied during traversal and an adaptive beam (`baseEf / selectivity`, capped). BM25 tokenization normalizes Unicode, folds accents ("ação" matches "acao") and removes English/Portuguese stopwords. Fusion is reciprocal rank fusion (Cormack et al., 2009) or min-max normalized weighted score fusion.

## Database, persistence and durability

`FaroDb` is the database layer around the indexes. One call opens (or creates) a `.faro` checkpoint plus its `.faro.wal` log; `FaroDb.InMemory()` gives the same API with no files.

```csharp
using Faro;

using var db = FaroDb.Open("knowledge.faro", new FaroOptions
{
    Durability = Durability.Batched, // Always | Batched (default) | None
});

var docs = db.GetOrCreateCollection("docs", new CollectionOptions<Doc>
{
    Dimensions = 768,
    Distance = Distance.Cosine,
    Index = new HnswOptions { M = 16, EfConstruction = 200, EfSearch = 64 },
    Quantization = Quantization.Int8(rescore: 4),
    Schema = schema,                                  // PayloadSchema<Doc>
    TypeInfo = DocContext.Default.Doc,                // source-generated JsonTypeInfo<Doc>
    FullTextField = d => d.Text,
});

docs.Upsert("doc-1", embedding, doc);                 // string keys; long, Guid and int also work
docs.UpsertBatch(items);                              // parallelised HNSW insertion
await docs.UpsertAsync("doc-2", embedding, doc);      // async variants accept CancellationToken

SearchHit<string>[] hits = docs.Search(queryEmbedding, k: 10, filter: Filter.Eq("tenant", "acme"));
HybridHit<string>[] hybrid = docs.HybridSearch(queryEmbedding, "ação", k: 10);

db.Checkpoint();                                      // temp file + atomic rename; resets the WAL
```

Open replays the WAL and truncates a torn tail, so a crash leaves a consistent prefix of committed operations. Vector pages of the checkpoint are memory-mapped; vectors added after the last checkpoint are held in arrays. Checkpoints are byte-deterministic for the same operation log, and the full byte layout is documented in [docs/file-format.md](docs/file-format.md). Do not search a database instance while checkpointing it.

## Distances

| Metric | API | Ordering | Notes |
| --- | --- | --- | --- |
| Cosine | `Distances.Cosine` / `CosineScalar` | lower is closer | `1 - cos(a,b)`, clamped to `[0,2]`; zero vectors treated as orthogonal |
| Dot product | `Distances.Dot` / `DotScalar` | higher is closer | raw inner product; use pre-normalized embeddings for cosine-like behaviour |
| Squared L2 | `Distances.EuclideanSquared` / `EuclideanSquaredScalar` | lower is closer | square root omitted; monotonic, so rankings are identical |
| Hamming | `Distances.Hamming` / `HammingScalar` | lower is closer | sign-bit packing (1 bit/dimension) + popcount; see `BitVectors` |

`Distances.Normalize` / `CopyNormalized` provide unit-length preprocessing. Every metric picks the widest accelerated path available (`Vector512` → `Vector256` → `Vector128` → scalar) and exposes explicit per-width kernels for tests; the scalar twins accumulate in `double` and act as the reference implementation.

## Internals

- `Faro.Internal.DistanceHeap` — a pooled binary heap over `(priority, id)` with four total orders. Searches rent the *inverse* of their metric order so the root is the worst kept hit and `PushBounded` evicts the correct item even when scores tie at the k boundary.
- `Faro.Internal.VisitedSet` — a pooled, generation-stamped visited set: `Reset()` starts a new generation in O(1), so graph searches never clear an array per query.

Both are rented from `ArrayPool<T>` and returned on `Dispose()`.

## Build and test

```bash
dotnet build Faro.sln -c Release -warnaserror
dotnet test Faro.sln -c Release
```

The suite runs on both `net8.0` and `net10.0` with xUnit v3 on Microsoft.Testing.Platform: distance equivalence across all SIMD widths, pooled heap/visited-set invariants, Flat exactness against naive scalar brute force, HNSW degree/reachability/tombstone invariants, deterministic builds, concurrent insert/search/delete stress, and zero-allocation assertions. Recall gates for Flat and HNSW print a recall/latency table via `tests/Faro.RecallTests`.

## Benchmarks

`benchmarks/Faro.Benchmarks` contains BenchmarkDotNet suites for raw kernels and Flat search (sequential, parallel, naive brute force). Run them on the target machine:

```bash
dotnet run -c Release -f net10.0 --project benchmarks/Faro.Benchmarks -- --filter '*'
```

No benchmark numbers are published here yet: the table will be filled in from an actual run on documented hardware once all milestones are complete.

## Native AOT

The sample publishes with Native AOT and no trim/AOT warnings:

```bash
dotnet publish samples/Faro.Samples.Rag -c Release -f net10.0 -r <rid> -p:PublishAot=true
```

CI performs this publish and runs the produced binary on `linux-x64`.

## Repository layout

```
src/Faro/                    core: distances, Flat index, pooled search structures
src/Faro.VectorData/         Microsoft.Extensions.VectorData connector (planned)
tests/Faro.Tests/            unit, equivalence, concurrency and allocation tests
tests/Faro.RecallTests/      recall/quality gates (Flat ground truth)
benchmarks/Faro.Benchmarks/  BenchmarkDotNet suites
samples/Faro.Samples.Rag/    offline markdown RAG sample
docs/                        architecture, file format, tuning
```

## License

MIT — see [LICENSE](LICENSE).
