using Faro.Samples.Rag;
using Xunit;

namespace Faro.Samples.Rag.Tests;

public sealed class SampleTests
{
    [Fact]
    public void HashingEmbedder_isDeterministicNormalizedAndSized()
    {
        var embedder = new HashingEmbedder(128);

        float[] first = embedder.Embed("hnsw graph search");
        float[] second = embedder.Embed("hnsw graph search");
        float[] other = embedder.Embed("bm25 full text ranking");

        Assert.Equal(128, first.Length);
        Assert.Equal(first, second);
        Assert.NotEqual(first, other);
        double norm = Math.Sqrt(first.Sum(value => (double)value * value));
        Assert.InRange(norm, 0.999, 1.001);
    }

    [Fact]
    public void HashingEmbedder_rejectsInvalidDimensions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HashingEmbedder(0));
        Assert.Throws<ArgumentNullException>(() => new HashingEmbedder(8).Embed(null!));
    }

    [Fact]
    public void MarkdownChunker_splitsBlocksStripsSyntaxAndDropsTinyFragments()
    {
        string markdown = """
            # Title

            This is a substantial paragraph with enough words to become a chunk.

            - short

            Another substantial paragraph that should survive the minimum length filter.
            """;

        List<string> chunks = MarkdownChunker.Chunk(markdown);

        Assert.Equal(2, chunks.Count);
        Assert.All(chunks, chunk => Assert.DoesNotContain('#', chunk));
        Assert.All(chunks, chunk => Assert.DoesNotContain("- short", chunk));
        Assert.Contains("substantial paragraph", chunks[0]);
    }

    [Fact]
    public void MarkdownChunker_treatsCrLfAndLfIdentically()
    {
        const string lf = """
            # Title

            First substantial paragraph with enough words to become a chunk.

            Second substantial paragraph with enough words to become a chunk too.
            """;
        string normalizedLf = lf.Replace("\r\n", "\n");
        string crlf = normalizedLf.Replace("\n", "\r\n");

        Assert.Equal(MarkdownChunker.Chunk(normalizedLf), MarkdownChunker.Chunk(crlf));
    }

    [Fact]
    public void MarkdownChunker_handlesEmptyAndLongBlocks()
    {
        Assert.Empty(MarkdownChunker.Chunk(string.Empty));
        Assert.Empty(MarkdownChunker.Chunk("# heading\n\n- x\n"));
        Assert.Throws<ArgumentNullException>(() => MarkdownChunker.Chunk(null!));

        string longBlock = string.Join(' ', Enumerable.Repeat("quantization compresses vectors with bytes", 40));
        List<string> chunks = MarkdownChunker.Chunk(longBlock);
        Assert.True(chunks.Count > 1);
        Assert.All(chunks, chunk => Assert.True(chunk.Length <= 600));
    }

    [Fact]
    public void RagPipeline_hybridSearchSurfacesKeywordAndVectorMatches()
    {
        using var pipeline = new RagPipeline();
        pipeline.Index(SampleDocuments.All);

        Assert.True(pipeline.Count >= SampleDocuments.All.Count);

        IReadOnlyList<RagSearchResult> results = pipeline.Search(
            "how can embeddings be compressed to one bit per dimension", 3);
        Assert.NotEmpty(results);
        Assert.Equal("quantization.md", results[0].Title);
        Assert.True(results[0].TextScore > 0);
        Assert.True(results[0].VectorScore > 0);
        Assert.True(results[0].Score > results[^1].Score);
    }

    [Fact]
    public void RagPipeline_offCorpusQueryHasNoKeywordEvidence()
    {
        using var pipeline = new RagPipeline();
        pipeline.Index(SampleDocuments.All);

        IReadOnlyList<RagSearchResult> results = pipeline.Search("the great wall of china", 3);

        Assert.NotEmpty(results);
        Assert.All(results, result => Assert.Equal(0, result.TextScore));
    }

    [Fact]
    public void RagPipeline_isDeterministicAcrossInstances()
    {
        using var first = new RagPipeline();
        using var second = new RagPipeline();
        first.Index(SampleDocuments.All);
        second.Index(SampleDocuments.All);

        var query = "hnsw graph search";
        Assert.Equal(
            first.Search(query, 3).Select(result => result.Title).ToArray(),
            second.Search(query, 3).Select(result => result.Title).ToArray());
    }
}
