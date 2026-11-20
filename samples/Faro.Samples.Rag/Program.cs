namespace Faro.Samples.Rag;

/// <summary>
/// A miniature offline RAG pipeline: chunk markdown, embed with a deterministic local hashing
/// embedder, index in Faro and run vector + BM25 hybrid search. No network and no server.
/// </summary>
public static class Program
{
    private const int TopK = 3;

    public static int Main(string[] args)
    {
        IReadOnlyList<(string Title, string Markdown)> documents = LoadDocuments(args);
        using var pipeline = new RagPipeline();
        int indexed = pipeline.Index(documents);

        Console.WriteLine($"Indexed {indexed} chunks from {documents.Count} documents " +
                          "(256-d deterministic hashing embeddings, cosine distance, BM25 hybrid fusion).");
        Console.WriteLine();

        string[] queries =
        [
            "how does hnsw keep search fast on large collections",
            "what ranking formula does full text search use",
            "how can embeddings be compressed to one bit per dimension",
            "an embedded database that plugs into semantic kernel",
            "the great wall of china", // off-corpus query: retrieval should be weak
        ];

        foreach (string query in queries)
        {
            Console.WriteLine($"query: \"{query}\"");
            foreach (RagSearchResult result in pipeline.Search(query, TopK))
            {
                string preview = result.Text.Length <= 100 ? result.Text : result.Text[..100] + "…";
                Console.WriteLine($"  rrf {result.Score,6:F4}  vec {result.VectorScore,7:F4}  " +
                                  $"bm25 {result.TextScore,6:F3}  {result.Title,-22}  {preview}");
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
            foreach (string path in Directory.EnumerateFiles(args[0], "*.md"))
            {
                loaded.Add((Path.GetFileName(path), File.ReadAllText(path)));
            }

            loaded.Sort(static (left, right) => StringComparer.Ordinal.Compare(left.Item1, right.Item1));
            if (loaded.Count > 0)
            {
                return loaded;
            }

            Console.WriteLine($"no *.md files under '{args[0]}'; falling back to the built-in corpus.");
        }

        return SampleDocuments.All;
    }
}
