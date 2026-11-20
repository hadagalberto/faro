# Faro

**Faro** is an embedded, zero-server vector database for .NET — "SQLite for embeddings". One NuGet package, one file on disk, SIMD distance kernels, exact and approximate indexes, and a path into Semantic Kernel / Microsoft.Extensions.AI RAG pipelines. MIT licensed, targets `net8.0` and `net10.0`, and is built to be Native AOT and trimming compatible.

> **Status — Milestone 7:** Faro includes the `Microsoft.Extensions.VectorData` connector, an offline markdown RAG sample, exact/HNSW indexes, quantization with rescoring, typed payloads, filters, BM25 hybrid search, and a versioned single-file database with CRC32C WAL, torn-tail recovery, atomic checkpoints, memory mapping, in-memory mode and generic keys. This milestone adds measured BenchmarkDotNet results, the full documentation set and a stable zero-allocation search path. See [docs/architecture.md](docs/architecture.md) and [docs/tuning.md](docs/tuning.md).

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
| Microsoft.Extensions.VectorData connector (LINQ filters, hybrid) | ✅ | — |

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

## Microsoft.Extensions.VectorData

`Faro.VectorData` targets the latest stable `Microsoft.Extensions.VectorData.Abstractions` (11.x). It implements `VectorStore`, `VectorStoreCollection<TKey,TRecord>` and `IKeywordHybridSearchable<TRecord>`, so Faro can be used from Semantic Kernel / Microsoft.Extensions.AI pipelines. Record schemas come from the standard key/data/vector attributes or an explicit `VectorStoreCollectionDefinition`.

```csharp
using Faro.VectorData;
using Microsoft.Extensions.VectorData;

[JsonSerializable(typeof(Article))]
internal sealed partial class ArticleContext : JsonSerializerContext { }

public sealed class Article
{
    [VectorStoreKey] public string Id { get; set; } = "";
    [VectorStoreData(IsIndexed = true)] public string Tenant { get; set; } = "";
    [VectorStoreData(IsIndexed = true)] public int Year { get; set; }
    [VectorStoreData(IsIndexed = true)] public string[] Tags { get; set; } = [];
    [VectorStoreData(IsFullTextIndexed = true)] public string Text { get; set; } = "";
    [VectorStoreVector(768, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public float[] Vector { get; set; } = [];
}

using var store = FaroVectorStore.InMemory(new FaroVectorStoreOptions
{
    JsonSerializerContext = ArticleContext.Default,
});
var articles = store.GetCollection<string, Article>("articles");

await articles.UpsertAsync(article);
await foreach (var result in articles.SearchAsync(queryEmbedding, 10, new VectorSearchOptions<Article>
{
    Filter = a => a.Tenant == "acme" && a.Year >= 2020 && a.Tags.Contains("rag"),
}))
{
    Console.WriteLine(result.Record.Id);
}

var hybrid = (IKeywordHybridSearchable<Article>)articles;
await foreach (var result in hybrid.HybridSearchAsync(queryEmbedding, ["ação", "rag"], 10))
{
    Console.WriteLine(result.Record.Id);
}
```

The LINQ translator supports `&&`, `||`, `!`, equality/inequality, numeric ranges, bare bool fields and `Contains` on tag/keyword fields (in either direction), with captured values inlined by the abstractions' preprocessing visitor. Unsupported expressions throw `NotSupportedException` instead of being silently ignored.

`GetCollection` carries the abstractions' `RequiresDynamicCode` / `RequiresUnreferencedCode` annotations because attribute-based model building uses reflection. For fully trimmed applications use `GetDynamicCollection` with an explicit definition and a registered payload serializer.

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

Both are rented from Faro's private, non-trimming `FaroPool<T>` (`ArrayPool<T>.Create()`) and returned on `Dispose()`. The shared BCL pool is deliberately not used: its gen2 GC callback trims idle buffers, which would make a steady-state rent allocate after an unrelated collection.

## Build and test

```bash
dotnet build Faro.sln -c Release -warnaserror
dotnet test Faro.sln -c Release
```

The suite runs on both `net8.0` and `net10.0` with xUnit v3 on Microsoft.Testing.Platform: distance equivalence across all SIMD widths, pooled heap/visited-set invariants, Flat exactness against naive scalar brute force, HNSW degree/reachability/tombstone invariants, deterministic builds, concurrent insert/search/delete stress, and zero-allocation assertions. Recall gates for Flat and HNSW print a recall/latency table via `tests/Faro.RecallTests`.

## Benchmarks

Measured with BenchmarkDotNet 0.15.8 on this machine: **AMD Ryzen 9 7900X3D (4.40 GHz, 12 cores / 24 threads), 63 GB RAM, Windows 11 Pro (10.0.26300), .NET 10.0.14, X64 RyuJIT x86-64-v4, Concurrent Workstation GC, AVX-512 capable (Vector256 hardware vector size)**. Job: 2 warmup + 5 iterations. `Mean` is latency; the ops/s column is derived from it. Full reports are reproduced by the command below.

**Insert** — 5,000 unit vectors of 128 dimensions, one vector per operation (fresh index per invocation, so growth is included):

| Index | Mean / vector | ≈ vectors/s | Allocated / vector |
| --- | ---: | ---: | ---: |
| Flat | 109.1 ns | 9.17 M | 512 B |
| Flat + int8 (rescore 4) | 228.3 ns | 4.38 M | 1,092 B |
| Flat + binary (rescore 8) | 353.5 ns | 2.83 M | 980 B |
| HNSW (M=16, efC=128) | 669.5 µs | 1,494 | 203.9 KB |
| HNSW + int8 (rescore 4) | 654.7 µs | 1,527 | 204.5 KB |
| HNSW + binary (rescore 8) | 641.6 µs | 1,559 | 204.4 KB |

**Search** — 20,000 unit vectors of 128 dimensions, `k = 10`, Cosine, sequential Flat path, HNSW `M=16` / `EfConstruction=128` / `EfSearch=64`:

| Index | Mean / query | ≈ queries/s | Allocated / query |
| --- | ---: | ---: | ---: |
| Flat | 404.28 µs | 2,474 | 104 B |
| Flat + int8 + rescore(4) | 753.08 µs | 1,328 | 104 B |
| Flat + binary + rescore(8) | 147.52 µs | 6,778 | 104 B |
| HNSW | 148.92 µs | 6,714 | 104 B |
| HNSW + int8 + rescore(4) | 185.02 µs | 5,405 | 104 B |
| HNSW + binary + rescore(8) | 89.42 µs | 11,183 | 104 B |
| HNSW, buffered span overload | 150.69 µs | 6,636 | **0 B** |

The 104 B on the allocating overloads is exactly the returned `SearchResult[10]`; the buffered overload confirms zero steady-state allocation. Quantization speeds up the routing pass but pays for rescoring, so it is not always faster on 128-d data — its purpose is memory reduction and recall recovery (see the recall suite). On 768-d data the sequential Flat search measured 1,011 µs versus 368 µs in parallel, and 28,288 µs for a naive scalar sort-based brute force.

```bash
dotnet run -c Release -f net10.0 --project benchmarks/Faro.Benchmarks -- --filter '*'
```

Hardware, runtime and dataset shape must be restated whenever these numbers are compared. See [docs/tuning.md](docs/tuning.md) for the knobs behind them.

## Native AOT

The sample publishes with Native AOT and no trim/AOT warnings:

```bash
dotnet publish samples/Faro.Samples.Rag -c Release -f net10.0 -r <rid> -p:PublishAot=true
```

CI performs this publish with `-p:TrimmerSingleWarn=false -warnaserror` and runs the produced binary on both `linux-x64` and `win-x64`.

## Repository layout

```
src/Faro/                    core: distances, Flat/HNSW indexes, payloads, filtering, BM25, persistence
src/Faro.VectorData/         Microsoft.Extensions.VectorData.Abstractions connector
tests/Faro.Tests/            unit, equivalence, concurrency and allocation tests
tests/Faro.RecallTests/      recall/quality gates (Flat ground truth)
tests/Faro.VectorData.Tests/ end-to-end connector, LINQ filter and hybrid tests
tests/Faro.Samples.Rag.Tests/ embedder, chunker and RAG pipeline tests
benchmarks/Faro.Benchmarks/  BenchmarkDotNet suites
samples/Faro.Samples.Rag/    offline markdown RAG sample (hashing embedder + hybrid search)
docs/                        architecture, file format, tuning
```

## License

MIT — see [LICENSE](LICENSE).
