# Faro.VectorData

Connector over the [`Microsoft.Extensions.VectorData`](https://learn.microsoft.com/dotnet/api/microsoft.extensions.vectordata)
abstractions (`IVectorStore` / `IVectorStoreCollection<TKey, TRecord>` / keyword hybrid search)
so Faro can be used directly from Semantic Kernel and Microsoft.Extensions.AI pipelines.

**Status:** this project is scaffolded per the SPEC solution layout and carries the package
reference (`Microsoft.Extensions.VectorData.Abstractions`, latest stable) and the project
reference to `Faro`. The connector implementation lands with the SPEC §8 milestone
(VectorStore + VectorStoreCollection with record definitions, LINQ-expression-to-Faro filter
translation for the supported subset, and hybrid search where the abstraction provides it),
because it requires the payload/filter/hybrid search surface of the completed core
(collection schema, filters, BM25, persistence) to bind against — not a partial approximation.
