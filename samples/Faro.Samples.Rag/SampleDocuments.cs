namespace Faro.Samples.Rag;

/// <summary>Embedded sample corpus so the RAG sample runs fully offline.</summary>
public static class SampleDocuments
{
    public const string VectorDatabases = """
        # Vector databases

        A vector database stores embeddings: dense numeric vectors that represent the meaning of
        text, images or other data. Instead of exact keyword matching, retrieval scores the
        similarity between a query vector and every stored vector with a distance function such as
        cosine similarity, dot product or Euclidean distance.

        Exact search over millions of vectors is expensive, so production systems combine
        approximate nearest neighbour indexes with rescoring. The HNSW graph index answers a
        query in milliseconds by exploring only a small fraction of the collection, keeping
        recall high while cutting latency by orders of magnitude compared to brute force.

        Faro is an embedded vector database for .NET: one library, one file on disk, no server to
        operate. It packs SIMD distance kernels, graph indexes, quantization, filtered and hybrid
        search, and a Microsoft.Extensions.VectorData connector so it plugs directly into
        Semantic Kernel and Microsoft.Extensions.AI pipelines.
        """;

    public const string Hnsw = """
        # Hierarchical Navigable Small World graphs

        HNSW, introduced by Malkov and Yashunin, is the dominant approximate nearest neighbour
        index for dense vector search. It builds a hierarchy of proximity graphs: the top layers
        are sparse long-range links, while the bottom layer contains every vector with short
        links to its nearest neighbours.

        A query starts at the entry point of the top layer and greedily descends towards the
        query, narrowing the candidate set at every level. On the bottom layer a bounded beam
        search controlled by the efSearch parameter expands the most promising candidates. Larger
        efSearch values visit more nodes and raise recall at the cost of latency.

        Insertion links each new vector to its M closest neighbours per layer, chosen with a
        heuristic that keeps the graph well connected instead of simply taking the closest ones.
        The construction cost parameter efConstruction controls how much effort is spent
        building high quality links, which pays off later in search recall.
        """;

    public const string Bm25 = """
        # BM25 full-text ranking

        BM25, formalised by Robertson and Zaragoza, ranks documents by term frequency, inverse
        document frequency and length normalisation. A term that appears often in one document
        but rarely in the corpus is strong evidence of relevance; very long documents are
        discounted so that sheer verbosity does not dominate the score.

        Hybrid search fuses vector and keyword results because they fail differently: embeddings
        capture meaning but miss rare identifiers, while keyword search hits exact terms but
        misses paraphrases. Reciprocal Rank Fusion, described by Cormack, Clarke and Buettcher in
        2009, merges two ranked lists by summing inverse ranks, needing no score calibration
        between the two systems.
        """;

    public const string Quantization = """
        # Quantization for vector search

        Storing full float32 vectors is expensive: a single 768-dimension embedding costs three
        kilobytes. Scalar quantization maps every dimension to one byte using per-dimension
        ranges learned from a sample, cutting memory by four times with a small accuracy cost.
        Binary quantization goes further and keeps only the sign of every dimension, one bit per
        dimension, so distances reduce to popcount operations over packed words.

        Quantized vectors are used as a fast first pass: the index over-samples candidates with
        cheap quantized distances and then rescores the survivors against the original
        full-precision vectors. With a generous oversampling factor the second pass recovers
        nearly all of the lost recall while keeping most of the speed and memory benefit.
        """;

    public static IReadOnlyList<(string Title, string Markdown)> All { get; } =
    [
        ("vector-databases.md", VectorDatabases),
        ("hnsw.md", Hnsw),
        ("bm25.md", Bm25),
        ("quantization.md", Quantization),
    ];
}
