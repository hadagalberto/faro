# Faro — SPEC

**Faro** is an embedded, zero-server **vector database for .NET** — "SQLite for embeddings". One NuGet package, one file on disk,
HNSW + SIMD + quantization + filtered and hybrid (vector + BM25) search, crash-safe persistence, and a
`Microsoft.Extensions.VectorData` connector so it plugs into Semantic Kernel / Microsoft.Extensions.AI RAG pipelines.
Open source, MIT, "Hadagalberto Junior". Targets `net8.0` and `net10.0`, Native AOT and trimming compatible.

Existing .NET HNSW packages are bare indexes. Faro's differentiator is the complete database: quantization with rescoring,
filtered search that stays accurate under selective filters, hybrid search, WAL durability, concurrency, and measured recall.
This is NOT a toy: no stubs, no TODO, no NotImplementedException, no placeholder or skipped tests, no "simplified" algorithms.

## Solution layout

```
Faro.sln
src/Faro/                    core (deps allowed: System.Numerics.Tensors only)
src/Faro.VectorData/         Microsoft.Extensions.VectorData.Abstractions connector (latest stable)
tests/Faro.Tests/            xUnit (+ FsCheck where useful)
tests/Faro.RecallTests/      recall/quality suite (separate project, runs in CI with moderate sizes)
benchmarks/Faro.Benchmarks/  BenchmarkDotNet (insert/search throughput, latency, memory; vs brute force)
samples/Faro.Samples.Rag/    console RAG sample: chunk markdown files, embed with a deterministic local hashing embedder (no network), hybrid search
.github/workflows/ci.yml     build + test ubuntu/windows, net8+net10, AOT publish smoke test of a sample
Directory.Build.props        nullable, warnaserror, deterministic, SourceLink, package metadata, IsAotCompatible=true
README.md + README.pt-BR.md, docs/ (architecture.md, file-format.md, tuning.md), CHANGELOG.md, CONTRIBUTING.md, LICENSE
```

## Core features

1. **Distances**: Cosine (with optional pre-normalization), DotProduct, Euclidean (squared L2), Hamming (binary). Implemented with
   `System.Numerics.Tensors.TensorPrimitives` and/or explicit `Vector128/256/512` intrinsics with scalar fallback; tests prove all code paths
   produce identical results (within float tolerance) to a naive scalar reference.
2. **Indexes**
   - `Flat` exact index (SIMD brute force, parallel for large collections) — also the ground truth for recall tests.
   - `Hnsw` (Malkov & Yashunin 2016/2018): configurable `M`, `M0 = 2*M`, `efConstruction`, `efSearch`, level multiplier `1/ln(M)`,
     **heuristic neighbour selection (Algorithm 4, keepPrunedConnections + extendCandidates options)**, deterministic seed option,
     thread-safe concurrent inserts (fine-grained node locks) concurrent with searches, deletes via tombstones with neighbour repair, and
     `Compact()` that rebuilds links without deleted nodes. Visited-set pooling to avoid allocations on search.
3. **Quantization** (per collection, optional):
   - Scalar int8 (per-dimension min/max trained on a sample, or quantile-clipped), int8 SIMD distance.
   - Binary (sign bit, 1-bit/dim) with popcount Hamming.
   - Search on quantized vectors with **oversampling + rescoring** on full-precision vectors (configurable factor).
   Recall tests must show rescoring recovers recall (assert numbers).
4. **Filtering**: payload fields declared in the schema can be indexed (keyword/equality, numeric range, bool, tags/any-of).
   Filter API: `Filter.Eq("tenant", "acme") & Filter.Range("year", 2020, 2024) | Filter.In("tag", ...)` (composable, immutable).
   Execution planner: estimate selectivity from payload indexes; high selectivity (few matches) → exact search over the candidate set;
   otherwise → filtered HNSW traversal (filter evaluated during traversal, ef dynamically expanded) — tests assert recall ≥ 0.9 with a filter matching 1% of points.
5. **Hybrid search**: optional full-text field with an in-house BM25 inverted index (tokenizer with Unicode normalization, lowercasing,
   accent folding — so "ação" matches "acao" — pluggable stopwords for en/pt), and fusion via **Reciprocal Rank Fusion** (k configurable) or weighted score fusion.
6. **Persistence**: single-file database (`.faro`) with a documented, versioned binary format (`docs/file-format.md`): header with magic/version/checksums,
   pages for vectors, graph, payloads, text index. **Write-ahead log** for upserts/deletes with CRC32C per record, fsync policy configurable
   (`Always`, `Batched`, `None`), automatic recovery on open (replay WAL, truncate torn tail), checkpoint compaction into the main file using
   write-to-temp + atomic rename. Memory-mapped reads for vectors. In-memory mode (`FaroDb.InMemory()`).
   Crash tests: simulate torn writes / truncated WAL at random offsets → reopening yields a consistent DB containing a prefix of committed ops.
7. **API**
   ```csharp
   using var db = FaroDb.Open("knowledge.faro", new FaroOptions { Durability = Durability.Batched });
   var docs = db.GetOrCreateCollection<Doc>("docs", new CollectionOptions {
       Dimensions = 768, Distance = Distance.Cosine,
       Index = new HnswOptions { M = 16, EfConstruction = 200 },
       Quantization = Quantization.Int8(rescore: 4),
       FullTextField = d => d.Text });
   docs.Upsert(id, vector, doc);            // also UpsertBatch(IEnumerable<...>) parallelised
   var hits = docs.Search(queryVector, k: 10, filter: Filter.Eq("tenant", "acme"), efSearch: 64);
   var hybrid = docs.HybridSearch(queryVector, "texto da busca", k: 10, fusion: Fusion.Rrf(60));
   ```
   Payload serialisation via System.Text.Json source generation (AOT-safe; user supplies `JsonTypeInfo<T>` or a `JsonSerializerContext`).
   Generic keys: `Guid`, `long`, `string`. Async variants accept `CancellationToken`.
8. **Faro.VectorData**: implement the latest `Microsoft.Extensions.VectorData` abstractions (VectorStore / VectorStoreCollection with record definitions,
   key/data/vector attributes, filters translated from LINQ expressions to Faro filters for the supported subset, hybrid search interface if present).
   Integration test that runs the abstraction's typical usage end-to-end.

## Quality bar

- Recall suite (synthetic clustered Gaussian data, deterministic seeds; plus random uniform): HNSW recall@10 ≥ 0.95 at efSearch=64 for 50k × 128-d;
  int8 + rescore ≥ 0.95; binary + rescore(8) ≥ 0.85 on 768-d normalized data; filtered search recall assertions as above. Print a recall/latency table.
- Unit tests ≥ 300 meaningful cases: distances, heap/priority queues, HNSW invariants (degree bounds, symmetric-ish connectivity, no links to deleted nodes after compaction,
  reachability of all live nodes from entry point), quantizers, filters, BM25 scoring against hand-computed values, RRF, file format round-trips, WAL recovery, concurrency stress (parallel upserts + searches, no exceptions, consistent counts).
- Zero allocations per search in steady state for Flat and HNSW without payload materialisation (asserted with GC.GetAllocatedBytesForCurrentThread, allowing the result array).
- `dotnet build -warnaserror` clean; all tests green on net8.0 and net10.0; AOT publish of the sample succeeds with no trim/AOT warnings.
- README: what/why, comparison table vs plain HNSW libs (features only, no fabricated numbers), quick start, filtering, hybrid, quantization, durability modes,
  VectorData/Semantic Kernel usage, real benchmark table produced by running the benchmarks on this machine (state the hardware), tuning guide.

## Hard constraints

- Never run `git push`, `gh`, `dotnet nuget push`, or anything that publishes. Commit locally after each milestone (conventional commits). Stay inside this repository.
- Cite papers/algorithms in code comments (HNSW paper, BM25 Robertson/Zaragoza, RRF Cormack et al. 2009, CRC32C Castagnoli).
