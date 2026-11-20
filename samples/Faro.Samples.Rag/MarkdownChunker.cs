namespace Faro.Samples.Rag;

/// <summary>
/// Splits markdown text into retrieval-sized chunks: blocks are separated by blank lines,
/// oversized blocks are split further on sentence boundaries, tiny fragments are dropped.
/// </summary>
public static class MarkdownChunker
{
    private const int MaxChunkLength = 600;
    private const int MinChunkLength = 40;

    public static List<string> Chunk(string markdown)
    {
        ArgumentNullException.ThrowIfNull(markdown);

        var chunks = new List<string>();
        foreach (var rawBlock in markdown.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            var block = StripMarkdownSyntax(rawBlock);
            if (block.Length == 0)
            {
                continue;
            }

            if (block.Length <= MaxChunkLength)
            {
                AddIfSubstantial(chunks, block);
                continue;
            }

            foreach (var sentence in SplitSentences(block))
            {
                AddIfSubstantial(chunks, sentence);
            }
        }

        return chunks;
    }

    private static string StripMarkdownSyntax(string block)
    {
        var lines = block.Split('\n');
        var cleaned = new List<string>();
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0)
            {
                continue;
            }

            if (trimmed.StartsWith('#'))
            {
                trimmed = trimmed.TrimStart('#').Trim();
            }
            else if (trimmed.StartsWith('-') || trimmed.StartsWith('*'))
            {
                trimmed = trimmed.TrimStart('-', '*').Trim();
            }

            cleaned.Add(trimmed);
        }

        return string.Join(" ", cleaned);
    }

    private static IEnumerable<string> SplitSentences(string block)
    {
        var current = new System.Text.StringBuilder();
        int sentenceStart = 0;
        for (int i = 0; i < block.Length; i++)
        {
            if (block[i] is '.' or '!' or '?')
            {
                current.Append(block[sentenceStart..(i + 1)]);
                sentenceStart = i + 1;
                if (current.Length >= MaxChunkLength / 2)
                {
                    yield return current.ToString();
                    current.Clear();
                }
            }
        }

        if (sentenceStart < block.Length)
        {
            current.Append(block[sentenceStart..]);
        }

        if (current.Length > 0)
        {
            yield return current.ToString();
        }
    }

    private static void AddIfSubstantial(List<string> chunks, string chunk)
    {
        var trimmed = chunk.Trim();
        if (trimmed.Length >= MinChunkLength)
        {
            chunks.Add(trimmed);
        }
    }
}
