# Faro

**Faro** é um banco de dados vetorial embutido e sem servidor para .NET — "SQLite para embeddings". Um pacote NuGet, um arquivo em disco, kernels de distância SIMD, índice exato e (em marcos futuros) índices aproximados, além de um caminho para pipelines RAG com Semantic Kernel / Microsoft.Extensions.AI. Licença MIT, alvos `net8.0` e `net10.0`, compatível com Native AOT e trimming.

> **Status — Marco 1:** scaffold da solução, CI e base de busca exata concluídos: distâncias Cosine / Dot / L2 ao quadrado / Hamming com kernels SIMD e referência escalar, heaps min/max e conjuntos de visitados agrupados em pool, e `FlatIndex` com busca paralela determinística. HNSW, quantização, filtros, busca híbrida BM25, persistência com WAL e o conector VectorData estão planejados; veja [docs/architecture.md](docs/architecture.md).

## Por que Faro

A maioria das bibliotecas HNSW para .NET entrega apenas um índice. Faro visa o banco de dados completo: recall medido, quantização com rescore, filtros precisos mesmo com alta seletividade, busca híbrida vetor + BM25 e persistência crash-safe em arquivo único. O `FlatIndex` exato desta versão é a verdade-ground usada para validar os índices aproximados futuros.

| Recurso | Faro (hoje) | Biblioteca HNSW comum |
| --- | --- | --- |
| Índice exato Flat com SIMD e busca paralela | ✅ | — |
| Kernels SIMD + referência escalar com testes de equivalência | ✅ | às vezes |
| Heaps e conjuntos de visitados em pool (sem alocação por busca) | ✅ | — |
| HNSW com construção determinística e reparo de vizinhos | planejado | ✅ |
| Quantização escalar/binária com rescore | planejado | — |
| Filtros de payload com planejamento por seletividade | planejado | — |
| BM25 + fusão RRF | planejado | — |
| Persistência em arquivo único com WAL e recuperação | planejado | — |
| Conector Microsoft.Extensions.VectorData | planejado | — |

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
```

Os resultados vêm em ordem do melhor para o pior — score crescente para Cosine / Euclidean / Hamming, decrescente para DotProduct — com empates exatos desempatados por id crescente. A busca sequencial não aloca nada além do array retornado na sobrecarga com alocação.

## Distâncias

| Métrica | API | Ordem | Observações |
| --- | --- | --- | --- |
| Cosseno | `Distances.Cosine` / `CosineScalar` | menor é mais próximo | `1 - cos(a,b)` em `[0,2]`; vetor zero é ortogonal |
| Produto escalar | `Distances.Dot` / `DotScalar` | maior é mais próximo | produto interno bruto |
| L2 ao quadrado | `Distances.EuclideanSquared` / `EuclideanSquaredScalar` | menor é mais próximo | raiz omitida; ranqueamento idêntico |
| Hamming | `Distances.Hamming` / `HammingScalar` | menor é mais próximo | empacotamento por bit de sinal + popcount |

Cada métrica escolhe o caminho SIMD mais largo disponível (`Vector512` → `Vector256` → `Vector128` → escalar) e expõe kernels explícitos por largura para os testes.

## Build e testes

```bash
dotnet build Faro.sln -c Release -warnaserror
dotnet test Faro.sln -c Release
```

A suíte roda em `net8.0` e `net10.0` com xUnit v3 sobre Microsoft.Testing.Platform: equivalência das distâncias em todas as larguras SIMD, invariantes de heap/conjunto de visitados, exatidão do Flat contra força bruta escalar, equivalência paralelo/sequencial, stress de concorrência e asserções de zero alocação.

## Benchmarks

`benchmarks/Faro.Benchmarks` contém suítes BenchmarkDotNet para kernels e busca Flat. nenhum número é publicado aqui ainda: a tabela será preenchida com uma execução real em hardware documentado quando todos os marcos estiverem completos.

## Licença

MIT — veja [LICENSE](LICENSE).
