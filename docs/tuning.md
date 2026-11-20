# Tuning

## Choosing a metric

| Situation | Recommended metric |
| --- | --- |
| Embeddings from a model that already returns unit vectors | `Distance.DotProduct` or `Distance.Cosine` |
| Raw embeddings of varying magnitude where direction matters | `Distance.Cosine` (normalization on by default) |
| Geometry matters, e.g. image features with meaningful magnitude | `Distance.Euclidean` (squared L2) |
| Binary/sign embeddings | `Distance.Hamming` (1 bit per dimension, popcount) |

Cosine with `NormalizeOnAdd = true` (the default) stores unit vectors and scores queries as `1 - dot`, the cheapest exact cosine path. Set `NormalizeOnAdd = false` only when you need the stored vector verbatim via `TryGetVector`.

## FlatOptions

| Option | Default | Guidance |
| --- | --- | --- |
| `Dimensions` | required | Must match every vector added; validation throws on mismatch. |
| `Distance` | `Cosine` | See the table above. |
| `NormalizeOnAdd` | `null` → true for Cosine | Disable only for verbatim storage. Throws if set to true for a non-Cosine metric. |
| `InitialCapacity` | 1024 | Raise it when bulk-loading to avoid array-copy growth pauses. |
| `ParallelThreshold` | 4096 | Collections at/above this size use `Parallel.For`. Set to `int.MaxValue` to force sequential (useful for deterministic latency or benchmarking). Set to `1` to force parallel in tests. |

## Allocations

The span overload `Search(ReadOnlySpan<float>, int, Span<SearchResult>)` reuses a caller-owned result buffer and rents its heap/query scratch from Faro's private, non-trimming `FaroPool<T>` (built on `ArrayPool<T>.Create()`). Its steady-state allocation is zero for all four metrics, on both Flat and HNSW, with or without quantization. The allocating overload differs only by the returned result array. A gen2 collection does not change this: the private pool is never trimmed by the runtime.

## Determinism and tie-breaks

Both search paths use a total order on `(score, id)`, so sequential and parallel searches return identical sequences and repeated searches are byte-for-byte stable. When comparing against a naive reference, use the same metric convention (`Distances.LowerIsCloser`) and break ties by ascending id.

## Measuring

```bash
dotnet run -c Release -f net10.0 --project benchmarks/Faro.Benchmarks -- --filter '*DistanceKernels*'
dotnet run -c Release -f net10.0 --project benchmarks/Faro.Benchmarks -- --filter '*FlatSearch*'
```

BenchmarkDotNet pinning and hardware reporting make these numbers machine-specific; always state the CPU and runtime version when sharing results. Recall gates live in `tests/Faro.RecallTests` and print their own table.

## Quantization knobs

`Quantization.Int8(rescore, quantile)` trains per-dimension ranges; `quantile` (e.g. 0.995) clips outliers so the byte is spent on the bulk of the distribution. `Quantization.Binary(rescore)` trades one bit per dimension for a popcount first pass. `rescore` is the oversampling factor: larger values recover more full-precision recall at the cost of rescoring more candidates. On 768-d normalized data, `rescore = 4` for int8 and `rescore = 8` for binary clear the SPEC gates; raise them if your data is less clustered. Supply a pretrained `ScalarQuantizer`/`BinaryQuantizer` for reproducible ranges, otherwise the index trains once from the first vectors added.

## HNSWOptions

| Option | Default | Guidance |
| --- | --- | --- |
| `M` | 16 | Layer-0 degree bound is `2 * M`. Higher builds slower but improves recall on difficult data. |
| `EfConstruction` | 200 | Larger builds slower and more connected; 128–400 is typical. |
| `EfSearch` | 64 | The main recall/latency dial. Raise until recall stops improving; the recall suite gates at 64. |
| `Seed` | `null` | Set for reproducible levels and graphs; leave unset for randomized builds. |
| `ExtendCandidates` / `KeepPrunedConnections` / `Mc` | `true` / `false` / 1.0 | Algorithm 4 knobs: candidate expansion, pruned fallback and diversity relaxation. Tests in `HnswAlgorithm4OptionsTests` prove each toggle changes the graph. |
| `InitialCapacity` | 1024 | Pre-size for bulk loads to avoid node-array copies. |

## Durability

`FaroOptions.Durability` selects `Always` (fsync every WAL record), `Batched` (default; fsync on flush/checkpoint) or `None` (fastest, no crash guarantee). `db.Checkpoint()` compacts the WAL into a fresh checkpoint with an atomic rename; do not search a database instance while checkpointing it.
