# Changelog

All notable changes to Faro are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow Semantic Versioning.

## [Unreleased]

### Planned

- `Microsoft.Extensions.VectorData` connector and integration tests.

## [0.5.0] — Milestone 5

### Added

- `FaroDb` / `FaroDbCollection<TKey,TPayload>`: open or create a single-file database, `InMemory()` mode, generic keys (`int`, `long`, `Guid`, `string`), batch upsert, sync and async APIs with `CancellationToken`, `Flush`/`Checkpoint` and their async variants.
- Versioned `.faro` checkpoint format with header/page checksums and 64-byte-aligned pages; deterministic bytes for identical operation logs. See `docs/file-format.md`.
- CRC32C (Castagnoli) write-ahead log with per-record checksums and `Always` / `Batched` / `None` durability modes.
- Recovery on open: replay valid records in order, ignore and truncate a torn tail, and rebuild a connected HNSW graph from the stored vectors and levels.
- Checkpointing through a temporary file, fsync and atomic rename; WAL reset only after the rename is durable.
- Read-only memory mapping of checkpoint vector pages (float vectors and packed Hamming bits), with rebinding after a checkpoint.
- Persistence tests: CRC32C known vectors, byte/key codecs, checkpoint round-trips (all metrics, quantization, filters, hybrid), WAL-only replay, durability modes, in-memory mode, generic keys, async contracts, deterministic bytes and crash simulation at random truncation offsets proving prefix consistency.

## [0.3.0] — Milestone 3

### Added

- `ScalarQuantizer`: symmetric per-dimension int8 trained on a sample, with optional quantile clipping, encode/decode, and int8 SIMD dot/squared-distance kernels (`Vector512`/`Vector256`/`Vector128` + scalar reference).
- `BinaryQuantizer`: 1-bit sign codes packed 64 per word with popcount Hamming distances; shared with the existing `BitVectors`/`BitKernels` helpers.
- `Quantization` factory with configurable oversampling/rescore factor, pretrained codecs and lazy training from the vectors added before the first search.
- Quantization integrated into `FlatIndex` and `HnswIndex`: codes steer the first candidate pass, oversampled survivors are rescored against full-precision vectors, and returned scores are always full precision. Graph construction stays full precision.
- Quantizer unit tests (trained ranges, quantile clipping, encode/decode extremes, hand-computed Hamming/bit codes, SIMD path equivalence, invalid-input contracts), quantized index integration/zero-allocation tests, and recall gates: int8 + rescore(4) ≥ 0.95 and binary + rescore(8) ≥ 0.85 on 768-d normalized data for both Flat and HNSW.

## [0.2.0] — Milestone 2

### Added

- `HnswIndex` / `HnswOptions`: layered navigable small-world graph with configurable `M` (`M0 = 2*M`), `EfConstruction`, `EfSearch`, level multiplier `1/ln(M)`, deterministic seed and Algorithm 4 neighbour selection with `ExtendCandidates`, `KeepPrunedConnections` and diversity factor `Mc`.
- Fine-grained per-node locking for concurrent inserts concurrent with lock-free searches over copy-on-write adjacency snapshots; pooled heaps, visited stamps and candidate buffers keep the span search overload allocation-free.
- Tombstone deletes with neighbour repair, plus `Compact()` that rebuilds the graph without deleted nodes while preserving ids and levels. A reciprocal spine keeps every layer reachable from the entry point even where heuristic pruning would otherwise strand a node.
- Invariant, delete/compaction, degree/reachability, concurrency-stress, zero-allocation and deterministic-build tests; recall suite gate for HNSW at mean recall@10 >= 0.95 with efSearch=64 on 50k x 128-d clustered data against Flat ground truth.

## [0.1.0] — Milestone 1

### Added

- Solution scaffold: `src/Faro`, `src/Faro.VectorData`, `tests/Faro.Tests`, `tests/Faro.RecallTests`, `benchmarks/Faro.Benchmarks`, `samples/Faro.Samples.Rag`, `Directory.Build.props`, `.editorconfig`, MIT `LICENSE`, GitHub Actions CI.
- Distance API: `Cosine`, `Dot`, squared `Euclidean` and `Hamming`, each with a SIMD dispatcher (`Vector512`/`Vector256`/`Vector128`), explicit per-width kernels, and double-precision scalar reference implementations.
- Bit-vector sign packing/unpacking helpers (`BitVectors`).
- Pooled `DistanceHeap` (four deterministic total orders, bounded top-k insertion) and generation-stamped `VisitedSet`.
- `FlatIndex`: exact search with SIMD scoring, deterministic parallel search, zero-allocation span overload, concurrent searches with a single writer, and input/option validation.
- 762 test cases per target framework (net8.0 + net10.0) covering kernel equivalence, heap/set invariants, exactness against naive brute force, parallel determinism, concurrency and steady-state allocations.
- Documentation: `README.md`, `README.pt-BR.md`, architecture, file-format design, tuning guide and contributing guide.

### Fixed

- Small-dimension cosine searches no longer pass oversized pooled buffers to `CopyNormalized`.
- `VisitedSet.Dispose` no longer returns caller-owned arrays to `ArrayPool<T>`.
