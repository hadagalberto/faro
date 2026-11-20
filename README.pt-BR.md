# Faro

**Faro** é um banco de dados vetorial embutido e sem servidor para .NET — "SQLite para embeddings". Um pacote NuGet, um arquivo em disco, kernels de distância SIMD, índices exato e aproximado (HNSW), quantização, filtros, busca híbrida e persistência crash-safe. Licença MIT, alvos `net8.0` e `net10.0`, compatível com Native AOT e trimming. Autor: Hadagalberto Junior.

> **Status — Marco 7:** inclui o conector `Microsoft.Extensions.VectorData`, o exemplo RAG offline com embedder determinístico local, índices Flat/HNSW, quantização com rescore, payloads tipados, filtros, BM25 híbrido, banco em arquivo único com WAL CRC32C, recuperação de write truncado, checkpoints atômicos, memória mapeada, modo in-memory, chaves genéricas, benchmark medido e documentação completa. Veja [docs/architecture.md](docs/architecture.md) e [docs/tuning.md](docs/tuning.md).

## Por que Faro

A maioria das bibliotecas HNSW para .NET entrega apenas um índice. Faro visa o banco completo: recall medido, quantização com rescore, filtros precisos mesmo com alta seletividade, busca híbrida vetor + BM25 e persistência crash-safe em arquivo único. O `FlatIndex` exato é a verdade-ground usada para validar os índices aproximados.

| Recurso | Faro (hoje) | Biblioteca HNSW comum |
| --- | --- | --- |
| Índice exato Flat com SIMD e busca paralela | ✅ | — |
| Kernels SIMD + referência escalar com testes de equivalência | ✅ | às vezes |
| Heaps e conjuntos de visitados em pool (sem alocação por busca) | ✅ | — |
| HNSW com construção determinística, inserts concorrentes e reparo de vizinhos | ✅ | ✅ |
| Quantização int8/binária com clipping por quantil e rescore | ✅ | — |
| Filtros de payload com planejamento por seletividade | ✅ | — |
| BM25 + fusão RRF | ✅ | — |
| Persistência em arquivo único com WAL e recuperação | ✅ | — |
| Conector Microsoft.Extensions.VectorData (filtros LINQ, híbrido) | ✅ | — |

## Início rápido

```csharp
using Faro;

var index = new FlatIndex(new FlatOptions
{
    Dimensions = 768,
    Distance = Distance.Cosine, // vetores normalizados na inserção por padrão
    ParallelThreshold = 4096,
});

int id = index.Add(embedding);
SearchResult[] hits = index.Search(queryEmbedding, k: 10);

// Estado estacionário sem alocação, com buffer do chamador:
var buffer = new SearchResult[10];
int written = index.Search(queryEmbedding, k: 10, buffer);
```

Resultados vêm do melhor para o pior — score crescente para Cosine / Euclidean / Hamming, decrescente para DotProduct — com empates exatos desempatados por id crescente.

## Busca aproximada (HNSW)

```csharp
var index = new HnswIndex(new HnswOptions
{
    Dimensions = 768,
    M = 16,               // grau máximo na camada 0 é 2 * M
    EfConstruction = 200,
    EfSearch = 64,
    Seed = 12345,         // omita para níveis aleatórios
});

int id = index.Add(embedding);
SearchResult[] hits = index.Search(queryEmbedding, k: 10);
SearchResult[] wider = index.Search(queryEmbedding, k: 10, efSearch: 128);
index.Delete(id);   // tombstone + reparo de vizinhos
index.Compact();    // reconstrói sem deletados, preservando ids
```

`ExtendCandidates`, `KeepPrunedConnections` e `Mc` controlam o Algoritmo 4. Inserts rodam em paralelo entre si e com buscas. Veja os testes de invariantes e recall em `tests/Faro.Tests` e `tests/Faro.RecallTests`.

## Quantização

A busca pontua códigos compactos primeiro, mantém `k * rescore` candidatos e então faz rescore contra os vetores originais em precisão total antes de retornar os melhores `k`. O score retornado é sempre o de precisão total.

```csharp
var scalar = ScalarQuantizer.Train(amostra, Dimensions, Distance.Cosine, quantile: 0.995);
var index = new HnswIndex(new HnswOptions
{
    Dimensions = 768,
    Quantization = Quantization.Int8(scalar, rescore: 4),
});

// Treino sob demanda com os primeiros vetores adicionados:
var flat8 = new FlatIndex(new FlatOptions
{
    Dimensions = 768,
    Quantization = Quantization.Int8(rescore: 4, quantile: 0.995),
});
var binary = new FlatIndex(new FlatOptions
{
    Dimensions = 768,
    Quantization = Quantization.Binary(rescore: 8),
});
```

`ScalarQuantizer` treina `min`/`max` por dimensão (ou limites com clipping por quantil) e codifica um byte por dimensão; `BinaryQuantizer` empacota sinais 64 por palavra e usa popcount Hamming. A construção do grafo sempre usa precisão total.

## Payloads, filtros e busca híbrida

`FaroCollection<T>` associa um índice a payloads tipados. A serialização usa `JsonTypeInfo<T>` source-generated (sem reflexão, AOT-safe).

```csharp
using Faro.Filtering;
using Faro.Hybrid;
using Faro.Payload;
using System.Text.Json.Serialization;

[JsonSerializable(typeof(Doc))]
internal sealed partial class DocContext : JsonSerializerContext { }

var schema = new PayloadSchema<Doc>()
    .Keyword("tenant", d => d.Tenant)
    .Numeric("year", d => d.Year)
    .Tags("tags", d => d.Tags)
    .FullText("text", d => d.Text);

var docs = new FaroCollection<Doc>(
    schema,
    new SourceGenPayloadSerializer<Doc>(DocContext.Default.Doc),
    new HnswOptions { Dimensions = 768, Distance = Distance.Cosine });

int id = docs.Add(embedding, doc);
Filter filter = (Filter.Eq("tenant", "acme") & Filter.Range("year", 2020, 2024))
                | Filter.In("tags", "ai", "rag");
SearchResult[] hits = docs.Search(queryEmbedding, k: 10, filter: filter);
SearchDecision decision = docs.LastDecision; // ExactCandidates ou FilteredTraversal

HybridResult[] hybrid = docs.HybridSearch(
    queryEmbedding, "ação", k: 10, fusion: Fusion.Rrf(60));
```

Filtros são imutáveis e combináveis com `&`, `|` e `!`. O planejador usa a seletividade estimada dos índices de payload: filtros estreitos são pontuados exatamente sobre os candidatos; filtros amplos traversam o grafo com o predicado aplicado durante a busca e `ef` adaptativo. O tokenizador do BM25 normaliza Unicode, remove acentos ("ação" casa com "acao") e stopwords en/pt. A fusão usa RRF (Cormack et al., 2009) ou combinação ponderada.

## Microsoft.Extensions.VectorData

`Faro.VectorData` implementa `VectorStore` / `VectorStoreCollection<TKey,TRecord>` e `IKeywordHybridSearchable<TRecord>` sobre `Microsoft.Extensions.VectorData.Abstractions` 11.0.0.

```csharp
using Faro.VectorData;
using Microsoft.Extensions.VectorData;

public sealed class Article
{
    [VectorStoreKey] public string Id { get; set; } = "";
    [VectorStoreData(IsIndexed = true)] public string Tenant { get; set; } = "";
    [VectorStoreData(IsFullTextIndexed = true)] public string Text { get; set; } = "";
    [VectorStoreVector(768, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public float[] Vector { get; set; } = [];
}

using var store = FaroVectorStore.InMemory(new FaroVectorStoreOptions
{
    JsonSerializerContext = ArticleContext.Default,
});
var articles = store.GetCollection<string, Article>("articles");
await articles.UpsertAsync(article);
await foreach (var result in articles.SearchAsync(queryEmbedding, 10, new VectorSearchOptions<Article>
{
    Filter = a => a.Tenant == "acme",
}))
{
    Console.WriteLine(result.Record.Id);
}
```

O tradutor LINQ cobre `&&`, `||`, `!`, igualdade, faixas numéricas, bool e `Contains` em campos de tag/keyword; expressões não suportadas lançam `NotSupportedException`. Para aplicações trimmed, use `GetDynamicCollection` com definição explícita.

## Banco de dados, persistência e durabilidade

```csharp
using var db = FaroDb.Open("knowledge.faro", new FaroOptions
{
    Durability = Durability.Batched, // Always | Batched (padrão) | None
});

var docs = db.GetOrCreateCollection<string, Doc>("docs", new CollectionOptions<Doc>
{
    Dimensions = 768,
    Distance = Distance.Cosine,
    Index = new HnswOptions { M = 16, EfConstruction = 200, EfSearch = 64 },
    Quantization = Quantization.Int8(rescore: 4),
    Schema = schema,
    TypeInfo = DocContext.Default.Doc,
    FullTextField = d => d.Text,
});

docs.Upsert("doc-1", embedding, doc);            // chaves int, long, Guid e string
docs.UpsertBatch(itens);                          // inserts HNSW paralelizados
await docs.UpsertAsync("doc-2", embedding, doc);  // variantes assíncronas com CancellationToken
SearchHit<string>[] hits = docs.Search(queryEmbedding, k: 10);
HybridHit<string>[] hybrid = docs.HybridSearch(queryEmbedding, "ação", k: 10);
db.Checkpoint();                                  // arquivo temporário + rename atômico
```

Na abertura o WAL é replayado e um write truncado no fim é descartado, deixando um prefixo consistente de operações confirmadas. As páginas de vetores do checkpoint são lidas via memory map. `FaroDb.InMemory()` oferece a mesma API sem arquivos. O formato binário está documentado em [docs/file-format.md](docs/file-format.md).

## Distâncias

| Métrica | API | Ordem | Observações |
| --- | --- | --- | --- |
| Cosseno | `Distances.Cosine` / `CosineScalar` | menor é mais próximo | `1 - cos(a,b)` em `[0,2]` |
| Produto escalar | `Distances.Dot` / `DotScalar` | maior é mais próximo | produto interno bruto |
| L2 ao quadrado | `Distances.EuclideanSquared` / `EuclideanSquaredScalar` | menor é mais próximo | raiz omitida; ranking idêntico |
| Hamming | `Distances.Hamming` / `HammingScalar` | menor é mais próximo | empacotamento por bit de sinal + popcount |

Cada métrica escolhe o caminho SIMD mais largo disponível (`Vector512` → `Vector256` → `Vector128` → escalar) e expõe kernels explícitos por largura para os testes.

## Internals e alocação

`DistanceHeap` e `VisitedSet` são `ref struct` sobre buffers de `FaroPool<T>`, um pool privado e sem trimming (`ArrayPool<T>.Create()`). Isso garante zero alocação em estado estacionário nas sobrecargas com buffer, mesmo após GCs de geração 2.

## Build e testes

```bash
dotnet build Faro.sln -c Release -warnaserror
dotnet test Faro.sln -c Release
```

A suíte roda em `net8.0` e `net10.0` com xUnit v3 sobre Microsoft.Testing.Platform: equivalência das distâncias, invariantes de heap/grafo, concorrência, alocação, recall (incluindo filtros a 1% de seletividade), serialização e recuperação de WAL.

## Benchmarks

Medido com BenchmarkDotNet 0.15.8 nesta máquina: **AMD Ryzen 9 7900X3D (4,40 GHz, 12 núcleos / 24 threads), 63 GB RAM, Windows 11 Pro (10.0.26300), .NET 10.0.14, X64 RyuJIT x86-64-v4, GC Concurrent Workstation, AVX-512 disponível**. Job: 2 warmups + 5 iterações. `Média` é a latência; ops/s é derivado dela.

**Insert** — 5.000 vetores unitários de 128 dimensões, um vetor por operação:

| Índice | Média / vetor | ≈ vetores/s | Alocado / vetor |
| --- | ---: | ---: | ---: |
| Flat | 109,1 ns | 9,17 M | 512 B |
| Flat + int8 (rescore 4) | 228,3 ns | 4,38 M | 1.092 B |
| Flat + binária (rescore 8) | 353,5 ns | 2,83 M | 980 B |
| HNSW (M=16, efC=128) | 669,5 µs | 1.494 | 203,9 KB |
| HNSW + int8 (rescore 4) | 654,7 µs | 1.527 | 204,5 KB |
| HNSW + binária (rescore 8) | 641,6 µs | 1.559 | 204,4 KB |

**Busca** — 20.000 vetores unitários de 128 dimensões, `k = 10`, Cosine, Flat sequencial, HNSW `M=16` / `EfConstruction=128` / `EfSearch=64`:

| Índice | Média / consulta | ≈ consultas/s | Alocado / consulta |
| --- | ---: | ---: | ---: |
| Flat | 404,28 µs | 2.474 | 104 B |
| Flat + int8 + rescore(4) | 753,08 µs | 1.328 | 104 B |
| Flat + binária + rescore(8) | 147,52 µs | 6.778 | 104 B |
| HNSW | 148,92 µs | 6.714 | 104 B |
| HNSW + int8 + rescore(4) | 185,02 µs | 5.405 | 104 B |
| HNSW + binária + rescore(8) | 89,42 µs | 11.183 | 104 B |
| HNSW, sobrecarga com buffer | 150,69 µs | 6.636 | **0 B** |

Os 104 B das sobrecargas com alocação são exatamente o `SearchResult[10]` retornado. Em 768 dimensões, a busca Flat sequencial mediu 1.011 µs contra 368 µs paralela e 28.288 µs para força bruta escalar ingênua.

```bash
dotnet run -c Release -f net10.0 --project benchmarks/Faro.Benchmarks -- --filter '*'
```

Sempre restate hardware, runtime e formato dos dados ao comparar esses números. Veja [docs/tuning.md](docs/tuning.md).

## Native AOT

```bash
dotnet publish samples/Faro.Samples.Rag -c Release -f net10.0 -r <rid> -p:PublishAot=true
```

O CI publica com `-p:TrimmerSingleWarn=false -warnaserror` e executa o binário em `linux-x64` e `win-x64`.

## Licença

MIT — veja [LICENSE](LICENSE).
