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

The span overload `Search(ReadOnlySpan<float>, int, Span<SearchResult>)` reuses a caller-owned result buffer and rents its heap/query scratch from `ArrayPool<T>`. Its steady-state allocation is zero for all four metrics. The allocating overload differs only by the returned result array.

## Determinism and tie-breaks

Both search paths use a total order on `(score, id)`, so sequential and parallel searches return identical sequences and repeated searches are byte-for-byte stable. When comparing against a naive reference, use the same metric convention (`Distances.LowerIsCloser`) and break ties by ascending id.

## Measuring

```bash
dotnet run -c Release -f net10.0 --project benchmarks/Faro.Benchmarks -- --filter '*DistanceKernels*'
dotnet run -c Release -f net10.0 --project benchmarks/Faro.Benchmarks -- --filter '*FlatSearch*'
```

BenchmarkDotNet pinning and hardware reporting make these numbers machine-specific; always state the CPU and runtime version when sharing results. Recall gates live in `tests/Faro.RecallTests` and print their own table.

## Planned knobs

HNSW will expose `M`, `EfConstruction`, `EfSearch`, a deterministic build seed, and neighbour-selection options; quantization will expose int8/binary modes with a rescoring oversampling factor; durability will expose `Always`/`Batched`/`None`. These are documented here as design intent and are not available in Milestone 1.
