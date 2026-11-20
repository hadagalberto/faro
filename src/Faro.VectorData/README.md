# Faro.VectorData

Connector for [`Microsoft.Extensions.VectorData.Abstractions`](https://www.nuget.org/packages/Microsoft.Extensions.VectorData.Abstractions)
11.0.0 (latest stable). It maps the abstractions' record model onto Faro collections so Faro can
be used from Semantic Kernel / Microsoft.Extensions.AI pipelines.

## What is implemented

- `FaroVectorStore` over an existing `FaroDb`, a `.faro` file, or an in-memory database.
- `FaroVectorStoreCollection<TKey,TRecord>` with CRUD, batch upsert, filtered retrieval, vector
  search (`SearchAsync`) and keyword hybrid search (`IKeywordHybridSearchable<TRecord>`).
- Schemas from `[VectorStoreKey]` / `[VectorStoreData]` / `[VectorStoreVector]` attributes or from
  an explicit `VectorStoreCollectionDefinition`; dynamic dictionary records are supported too.
- LINQ filter translation for `&&`, `||`, `!`, equality/inequality, numeric ranges, bare bool
  fields and `Contains` on tag/keyword fields (both directions). Captured variables are inlined by
  the abstractions' preprocessing visitor. Unsupported expressions throw `NotSupportedException`.
- AOT-safe payload persistence through a caller-supplied source-generated JSON context or explicit
  `IPayloadSerializer<TRecord>` registration.

See the root [README](../../README.md#microsoftextensionsvectordata) for a usage example and the
supported-subset notes. `GetCollection` carries the abstractions' `RequiresDynamicCode` /
`RequiresUnreferencedCode` annotations; trimmed applications should use `GetDynamicCollection` with
an explicit definition and a registered serializer.
