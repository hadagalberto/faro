# Faro — architecture

## Layers

| Layer | Project | Status | Responsibility |
| --- | --- | --- | --- |
| Core | `src/Faro` | complete | Metrics, exact + HNSW indexes, quantization, payload schema/indexes, filters, planner, BM25 hybrid search, versioned single-file persistence with WAL and memory-mapped vector reads |
| Connector | `src/Faro.VectorData` | complete | `Microsoft.Extensions.VectorData` record definitions, LINQ filter translation, hybrid search adapter |
| Quality gates | `tests/Faro.Tests`, `tests/Faro.RecallTests` | complete | Equivalence, invariants, concurrency, allocations, recall (quantized and filtered), file-format round-trips and torn-WAL crash simulation |
| Tooling | `benchmarks/Faro.Benchmarks`, `samples/Faro.Samples.Rag` | complete | Throughput/latency measurement and an offline end-to-end sample |

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

Every buffer is rented from `FaroPool<T>`, a private `ArrayPool<T>.Create()` instance. `ArrayPool<T>.Shared` was deliberately rejected: it registers a gen2 GC callback that trims idle buffers, so a rent after an unrelated collection can allocate a fresh array and break the steady-state guarantee. The private `ConfigurableArrayPool` never trims automatically and cannot be drained by other libraries. Buffer sizes and counts are bounded by the index configuration and every rent has a matching `finally` return, including parallel and filtered paths.

## FlatIndex

Storage is one flat array: `float[capacity * dimensions]` for real metrics, `ulong[capacity * wordsPerVector]` packed sign bits for Hamming. `Add` writes the vector, then publishes the count with a volatile write; readers do an acquire read and clamp to the capacity of the storage array they observe. That gives a single writer and lock-free concurrent searches over a consistent prefix, with no locks in the read path. Arrays only grow, never shrink.

`Search` picks a path by collection size:

- **sequential** — one pooled heap (plus a pooled normalized/packed query), no allocations in steady state;
- **parallel** — `Parallel.For` over id partitions with a private heap per partition, then a deterministic merge into a final heap. Partition buffers are rented and returned after the join.

Both paths produce identical `(id, score)` sequences because every heap uses a total order.

## HnswIndex

`HnswIndex` assigns each node an exponentially distributed level from `1/ln(M)` (or a fixed seed), then links it on every layer from its level down to 0. Construction runs Algorithm 5's greedy descent above the node's level and a beam search (`EfConstruction`) at each lower layer; Algorithm 4 selects diverse neighbours with optional candidate extension and a pruned-connection fallback. Nodes are scored with the same SIMD kernels as `FlatIndex` (including normalized-cosine `1 - dot` and packed Hamming).

Topology snapshots are captured under a short state lock; distance work and neighbour updates happen outside it. Each node has its own gate, adjacency arrays are copy-on-write, and locks are always taken in ascending id order, so inserts run concurrently with each other and with lock-free searches. Besides the bounded Algorithm 4 links, each node keeps a reciprocal spine edge to the closest surviving neighbour at each layer. The spine is exempt from pruning and is traversed by searches, which guarantees the live graph remains reachable from the entry point even in sparse regions.

`Delete` marks a tombstone, sweeps every node for references to it, repairs surviving neighbours by distance-ranked links and re-connects the spine. `Compact()` rebuilds the entire graph from live vectors at the same ids and levels, atomically publishing the new topology so in-flight searches finish against the old snapshot.

## Quantization

`ScalarQuantizer` trains per-dimension `min`/`max` (or quantile-clipped bounds) on a bounded sample and derives `scale = max(|min|,|max|)/127`; codes are signed bytes clamped to [-127,127]. `Int8Kernels` widen sbyte -> short -> int and multiply in SIMD lanes, with explicit 512/256/128 and scalar paths whose integer results are identical. `BinaryQuantizer` reuses `BitVectors.PackSignBits` and `BitKernels.Hamming` (popcount). Distances are returned in the collection metric's ordering convention.

An index with `Quantization` stores codes alongside full vectors. The first pass scores codes and keeps `k * rescore` candidates; the second pass scores those with the exact SIMD kernels and keeps `k`, so returned scores are full precision. HNSW construction never uses codes — quantization only replaces the routing scorer during search, with the beam widened to the oversampling window. A pretrained codec makes builds reproducible; otherwise training happens once on first search from the first vectors added.

## Payloads, filters and hybrid search (milestone 4)

`PayloadSchema<T>` declares keyword, numeric, bool, tag and one full-text field with delegate selectors. `SourceGenPayloadSerializer<T>` stores payloads with a caller-supplied `JsonTypeInfo<T>` obtained from a `JsonSerializerContext`, so no reflection is used. Each non-text field has an inverted index (`TermIndex`, `NumericIndex`, `BoolIndex`); all share the vector index's integer ids and are updated atomically with the payload record.

`Filter` is an immutable tree. `Evaluate` over the index snapshot yields the exact candidate id set (set intersection for `&`, union for `|`, complement for `!`), while `Matches` evaluates the same tree against one stored payload during traversal. `SearchPlanner` compares candidate fraction to `ExactThreshold`: at or below it the candidates are scored exactly with the same SIMD kernels as the indexes; above it the HNSW layer-0 search applies the predicate at visit time, seeds the beam with indexed ids, and widens `ef` to `min(MaxEf, ceil(baseEf / selectivity))`. Upper layers are unfiltered and only choose an entry point.

`Tokenizer` uses FormD decomposition and drops non-spacing marks to fold accents, lowercases with `char.ToLowerInvariant`, splits on non-letter/digit runes and removes pluggable en/pt stopwords. `Bm25Index` keeps term -> (doc -> tf) postings and document lengths; scores use Robertson/Zaragoza idf and tf saturation. `Fusion` merges vector and BM25 rankings with RRF (`1/(k+rank)`) or min-max normalized weighted scores, with ascending id as the final tie-break.

## Persistence (milestone 5)

`FaroDb` owns a serializer-independent `DatabaseState` that mirrors every committed collection, node and payload. Writes append a self-contained WAL record (CRC32C, Castagnoli), update the in-memory indexes and update that state. Opening reads the checkpoint, validates every page checksum, replays the WAL and truncates a torn tail. Checkpoints write a new file to `<path>.tmp`, flush, atomically rename it, reset the WAL and rebind the memory map. Stored vectors are read directly from the mapped page; vectors added since the last checkpoint use owned arrays. The in-memory graph is rebuilt from vectors and levels on open, because a WAL upsert cannot carry reciprocal link changes made to older nodes. Keys (`int`, `long`, `Guid`, `string`) are encoded independently of the payload serializer. See [file-format.md](file-format.md) for byte-level details.

## Verification

`tests/Faro.Tests` asserts kernel equivalence, heap/visited-set invariants, HNSW degree and reachability invariants, delete/compaction behaviour, concurrency stress, WAL crash prefixes and steady-state allocations. `tests/Faro.RecallTests` gates recall against Flat ground truth; `tests/Faro.VectorData.Tests` and `tests/Faro.Samples.Rag.Tests` exercise the connector and sample end to end. Allocation tests share the non-parallel `ZeroAllocation` collection and use a common harness that warms every path with the measurement parameters and forces a gen2 collection before measuring. BenchmarkDotNet suites in `benchmarks/Faro.Benchmarks` measure kernels, Flat/HNSW insert and Flat/HNSW search with and without quantization.

## References

- Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2016/2018).
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009).
- Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (2009).
- Warren, *Hacker's Delight* §5-1, for the SWAR popcount idiom.
- Castagnoli et al., "Optimization of Cyclic Redundancy-Check Codes with 24 and 32 Parity Bits" (1993), for CRC32C.
