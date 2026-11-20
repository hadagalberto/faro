using Faro;

namespace Faro.Samples.Rag;

/// <summary>
/// A miniature RAG pipeline: chunk markdown documents, embed each chunk with a deterministic
/// local hashing embedder (no network), index the embeddings in Faro and run exact semantic
/// search over them. Keyword/BM25 hybrid fusion arrives with the full-text search milestone;
/// this milestone exercises the storage and search path end to end.
/// </summary>
public static class Program
{
    private const int EmbeddingDimensions = 256;
    private const int TopK = 3;

    public static int Main(string[] args)
    {
        var documents = LoadDocuments(args);
        var embedder = new HashingEmbedder(EmbeddingDimensions);

        var index = new FlatIndex(new FlatOptions
        {
            Dimensions = EmbeddingDimensions,
            Distance = Distance.Cosine,
        });

        var chunkTitles = new List<string>();
        var chunkPreviews = new List<string>();
        foreach (var (title, markdown) in documents)
        {
            foreach (var chunk in MarkdownChunker.Chunk(markdown))
            {
                int id = index.Add(embedder.Embed(chunk));
                while (chunkTitles.Count <= id)
                {
                    chunkTitles.Add(string.Empty);
                    chunkPreviews.Add(string.Empty);
                }

                chunkTitles[id] = title;
                chunkPreviews[id] = chunk.Length <= 110 ? chunk : chunk[..110] + "…";
            }
        }

        Console.WriteLine($"Indexed {chunkTitles.Count} chunks from {documents.Count} documents " +
                          $"({EmbeddingDimensions}-d hashing embeddings, {Distance.Cosine} distance).");
        Console.WriteLine();

        string[] queries =
        [
            "how does hnsw keep search fast on large collections",
            "what ranking formula does full text search use",
            "how can embeddings be compressed to one bit per dimension",
            "an embedded database that plugs into semantic kernel",
            "the great wall of china", // off-corpus query: farthest chunks rank last
        ];

        foreach (var query in queries)
        {
            Console.WriteLine($"query: \"{query}\"");
            var hits = index.Search(embedder.Embed(query), TopK);
            foreach (var hit in hits)
            {
                Console.WriteLine($"  score {hit.Score,7:F4}  {chunkTitles[hit.Id],20}  {chunkPreviews[hit.Id]}");
            }

            Console.WriteLine();
        }

        return 0;
    }

    private static IReadOnlyList<(string Title, string Markdown)> LoadDocuments(string[] args)
    {
        if (args.Length > 0 && Directory.Exists(args[0]))
        {
            var loaded = new List<(string, string)>();
            foreach (var path in Directory.EnumerateFiles(args[0], "*.md"))
            {
                loaded.Add((Path.GetFileName(path), File.ReadAllText(path)));
            }

            if (loaded.Count > 0)
            {
                return loaded;
            }

            Console.WriteLine($"no *.md files under '{args[0]}'; falling back to the built-in corpus.");
        }

        return SampleDocuments.All;
    }
}
