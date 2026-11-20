# Faro

**Faro** is an embedded, zero-server vector database for .NET — "SQLite for embeddings". One NuGet package, one file on disk, SIMD distance kernels, exact and approximate indexes, and a path into Semantic Kernel / Microsoft.Extensions.AI RAG pipelines. MIT licensed, targets `net8.0` and `net10.0`, and is built to be Native AOT and trimming compatible.

> **Status — Milestone 2:** the exact-search foundation (SIMD + scalar distances, pooled heaps/visited sets, `FlatIndex`) is joined by `HnswIndex`: Algorithm 4 neighbour selection, deterministic builds, concurrent inserts with fine-grained locks, tombstone deletes with neighbour repair and `Compact()`. Recall is measured against Flat on 50k x 128-d clustered data. Quantization, filtering, BM25 hybrid search, WAL persistence and the VectorData connector are designed but not shipped yet; see [docs/architecture.md](docs/architecture.md).

## Why Faro

Most .NET HNSW packages hand you a bare index. Faro aims at the complete database around it: measured recall, quantization with rescoring, filters that stay accurate under high selectivity, hybrid vector + BM25 search, and crash-safe single-file persistence. The exact `FlatIndex` in this release is the ground-truth engine every later approximate index is validated against.

| Feature | Faro (today) | Plain HNSW library (typical) |
| --- | --- | --- |
| Exact Flat index with SIMD kernels and parallel search | ✅ | — |
| SIMD + scalar reference kernels with equivalence tests | ✅ | sometimes |
| Pooled heaps / visited sets (no per-search allocations) | ✅ | — |
| HNSW with deterministic builds, concurrent inserts and neighbour repair | ✅ | ✅ |
| Scalar/binary quantization with rescoring | planned | — |
| Payload filters with selectivity-aware planning | planned | — |
| BM25 full-text + RRF hybrid search | planned | — |
| Single-file persistence with WAL + recovery | planned | — |
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
