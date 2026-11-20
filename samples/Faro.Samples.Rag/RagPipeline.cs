using Faro.Hybrid;
using Faro.Payload;

namespace Faro.Samples.Rag;

/// <summary>
/// An offline RAG pipeline: deterministic hashing embeddings plus Faro's vector/BM25 hybrid
/// search over markdown chunks. No network, no model download, no server.
/// </summary>
public sealed class RagPipeline : IDisposable
{
    private const int DefaultDimensions = 256;
    private const int DefaultSeed = 12345;

    private readonly FaroDb _db;
    private readonly FaroDbCollection<int, Chunk> _chunks;
    private readonly HashingEmbedder _embedder;

    /// <summary>Creates an in-memory pipeline over the built-in or caller-supplied documents.</summary>
    /// <param name="dimensions">Hashing-embedder dimensionality.</param>
    /// <param name="seed">HNSW construction seed, fixed by default for reproducible output.</param>
    public RagPipeline(int dimensions = DefaultDimensions, int seed = DefaultSeed)
    {
        if (dimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions));
        }

        _embedder = new HashingEmbedder(dimensions);
        _db = FaroDb.InMemory();
        _chunks = _db.GetOrCreateCollection<int, Chunk>("chunks", new CollectionOptions<Chunk>
        {
            Dimensions = dimensions,
            Distance = Distance.Cosine,
            Index = new HnswOptions { M = 16, EfConstruction = 200, EfSearch = 64, Seed = seed },
            Schema = new PayloadSchema<Chunk>()
                .Keyword("title", chunk => chunk.Title)
                .FullText("text", chunk => chunk.Text),
            TypeInfo = RagJsonContext.Default.Chunk,
        });
    }

    /// <summary>Number of indexed chunks.</summary>
    public int Count => _chunks.Count;

    /// <summary>Chunks markdown documents into indexed passages, in a deterministic order.</summary>
    /// <param name="documents">Document titles and markdown bodies.</param>
    /// <returns>Number of chunks added.</returns>
    public int Index(IEnumerable<(string Title, string Markdown)> documents)
    {
        ArgumentNullException.ThrowIfNull(documents);
        int added = 0;
        foreach ((string title, string markdown) in documents)
        {
            foreach (string chunk in MarkdownChunker.Chunk(markdown))
            {
                _chunks.Upsert(added, _embedder.Embed(chunk), new Chunk { Title = title, Text = chunk });
                added++;
            }
        }

        return added;
    }

    /// <summary>Runs vector + BM25 hybrid search over the indexed chunks.</summary>
    /// <param name="query">The natural-language query.</param>
    /// <param name="top">Maximum number of results.</param>
    /// <returns>Results ordered by reciprocal-rank fusion score, best first.</returns>
    public IReadOnlyList<RagSearchResult> Search(string query, int top)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(top);

        HybridHit<int>[] hits = _chunks.HybridSearch(
            _embedder.Embed(query), query, top, Fusion.Rrf());
        var results = new List<RagSearchResult>(hits.Length);
        foreach (HybridHit<int> hit in hits)
        {
            if (_chunks.TryGet(hit.Key, out Chunk? chunk) && chunk is not null)
            {
                results.Add(new RagSearchResult(
                    chunk.Title,
                    chunk.Text,
                    hit.Score,
                    hit.VectorScore ?? 0,
                    hit.TextScore ?? 0));
            }
        }

        return results;
    }

    /// <inheritdoc />
    public void Dispose() => _db.Dispose();
}
