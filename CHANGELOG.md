# Changelog

All notable changes to Faro are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow Semantic Versioning.

## [Unreleased]

### Planned

- HNSW index with deterministic builds, concurrent inserts, tombstones and compaction.
- Scalar/binary quantization with oversampling and full-precision rescoring.
- Composable payload filters with selectivity-aware execution.
- BM25 full-text index and vector/keyword fusion (RRF, weighted scores).
- Single-file `.faro` persistence with WAL and crash recovery.
- `Microsoft.Extensions.VectorData` connector and integration tests.

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
