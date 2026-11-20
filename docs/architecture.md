# Faro — architecture

## Layers

| Layer | Project | Status | Responsibility |
| --- | --- | --- | --- |
| Core | `src/Faro` | Milestone 1 | Metrics, storage, exact index, pooled search structures |
| Connector | `src/Faro.VectorData` | planned | `Microsoft.Extensions.VectorData` record definitions, LINQ filter translation, hybrid search adapter |
| Quality gates | `tests/Faro.Tests`, `tests/Faro.RecallTests` | Milestone 1 | Equivalence, invariants, concurrency, allocations, recall |
| Tooling | `benchmarks/Faro.Benchmarks`, `samples/Faro.Samples.Rag` | Milestone 1 | Throughput/latency measurement and an offline end-to-end sample |

The core is intentionally dependency-free (only `System.Numerics.Tensors` is allowed by SPEC, and it is not needed yet). This keeps Native AOT and trimming predictable.

## Metric convention

`Distance` is either a distance (lower is closer: Cosine, Euclidean, Hamming) or a similarity (higher is closer: DotProduct). `Distances.LowerIsCloser(distance)` is the single source of truth for ordering decisions. All public results are best-first with ascending id as the final tie-break, so they are deterministic across CPU vendors and thread counts.

## Distance kernels

`VectorKernels` and `BitKernels` each expose four explicit paths — `Vector512`, `Vector256`, `Vector128`, scalar — selected by `IsHardwareAccelerated` at runtime. The dispatcher contains no virtual calls; the scalar twins accumulate in `double` and double as the reference implementations used by both the equivalence tests and the naive recall ground truth. Remainders after the last full SIMD block are handled by a scalar tail loop.

## Pooled search structures

`DistanceHeap` is a `ref struct` backed by `ArrayPool<(float,int)>`. Its four orders are the cross product of priority direction (min/max) and id tie direction (ascending/descending). The heap root is the maximum of the order. An index rents the **inverse** of its metric order, which makes the root the *worst* kept hit:

- lower-is-closer metrics rent `MaxIdDescending` (root = largest score, ties = highest id),
- higher-is-closer metrics rent `MinIdDescending` (root = smallest score, ties = highest id).

`PushBounded` replaces the root only when a candidate is worse in that order, i.e. better by the metric. Using the exact inverse including the tie direction is what keeps the lower id when equal scores straddle the k boundary.

`VisitedSet` is a `ref struct` over a pooled `int[]` of generation stamps. `Reset()` increments a generation instead of clearing the array, so O(1) per search; at `int.MaxValue` it clears once and restarts at 1. Its internal constructor accepts a caller-owned array for the overflow test; `Dispose()` returns only pool-rented arrays.

## FlatIndex

Storage is one flat array: `float[capacity * dimensions]` for real metrics, `ulong[capacity * wordsPerVector]` packed sign bits for Hamming. `Add` writes the vector, then publishes the count with a volatile write; readers do an acquire read and clamp to the capacity of the storage array they observe. That gives a single writer and lock-free concurrent searches over a consistent prefix, with no locks in the read path. Arrays only grow, never shrink.

`Search` picks a path by collection size:

- **sequential** — one pooled heap (plus a pooled normalized/packed query), no allocations in steady state;
- **parallel** — `Parallel.For` over id partitions with a private heap per partition, then a deterministic merge into a final heap. Partition buffers are rented and returned after the join.

Both paths produce identical `(id, score)` sequences because every heap uses a total order.

## Roadmap (design intent, not yet implemented)

1. **HNSW** — layered proximity graph, Algorithm 4 neighbour selection, fine-grained node locks for concurrent inserts, tombstones + `Compact()`, seeded builds.
2. **Quantization** — int8 with trained per-dimension ranges and binary sign quantization; oversampling + full-precision rescoring driven by measured recall.
3. **Filtering** — immutable composable filters over indexed payload fields; selectivity planner chooses exact scan over candidates vs filtered graph traversal.
4. **Hybrid** — Unicode-normalized/accent-folded tokenizer, BM25 inverted index (Robertson & Zaragoza), RRF fusion (Cormack et al. 2009).
5. **Persistence** — versioned single-file format with WAL (CRC32C, Castagnoli), configurable fsync, torn-tail recovery, checkpoint via temp file + atomic rename. See [file-format.md](file-format.md).
6. **Connector** — `Faro.VectorData` implements the current abstractions with source-generated payload serialization for AOT safety.

## References

- Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2016/2018).
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009).
- Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (2009).
- Warren, *Hacker's Delight* §5-1, for the SWAR popcount idiom.
- Castagnoli et al., "Optimization of Cyclic Redundancy-Check Codes with 24 and 32 Parity Bits" (1993), for CRC32C.
