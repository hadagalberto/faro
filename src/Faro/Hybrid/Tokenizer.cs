using System.Globalization;
using System.Text;

namespace Faro.Hybrid;

/// <summary>
/// Deterministic BM25 tokenizer: Unicode normalization, lowercasing, optional accent folding,
/// punctuation splitting and stopword removal.
/// </summary>
/// <remarks>
/// Accent folding uses canonical decomposition (FormD) and drops non-spacing marks, so "ação"
/// and "acao" produce the same token. Tokens are split at any character that is not a Unicode
/// letter or digit, which handles the Latin, Cyrillic, Greek, CJK and emoji cases without a
/// culture-dependent word breaker.
/// </remarks>
public sealed class Tokenizer
{
    /// <summary>Default tokenizer: English + Portuguese stopwords with accent folding.</summary>
    public static Tokenizer Default { get; } = new(Stopwords.EnglishAndPortuguese, foldAccents: true);

    private readonly Stopwords _stopwords;
    private readonly bool _foldAccents;

    /// <summary>Creates a tokenizer.</summary>
    /// <param name="stopwords">The stopword set to apply.</param>
    /// <param name="foldAccents">Whether to strip diacritics before stopword filtering.</param>
    /// <exception cref="ArgumentNullException"><paramref name="stopwords"/> is <see langword="null"/>.</exception>
    public Tokenizer(Stopwords stopwords, bool foldAccents = true)
    {
        ArgumentNullException.ThrowIfNull(stopwords);
        _stopwords = stopwords;
        _foldAccents = foldAccents;
    }

    /// <summary>Tokenizes text into normalized, stopword-free terms.</summary>
    /// <param name="text">The input text.</param>
    /// <returns>A new array of tokens, possibly empty.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <see langword="null"/>.</exception>
    public string[] Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0)
        {
            return [];
        }

        string normalized = text.Normalize(_foldAccents ? NormalizationForm.FormD : NormalizationForm.FormC);
        var tokens = new List<string>();
        var buffer = new StringBuilder(normalized.Length);

        foreach (char character in normalized)
        {
            UnicodeCategory category = CharUnicodeInfo.GetUnicodeCategory(character);
            if (char.IsLetterOrDigit(character))
            {
                buffer.Append(char.ToLowerInvariant(character));
                continue;
            }

            if (_foldAccents && category == UnicodeCategory.NonSpacingMark)
            {
                // Diacritic stripped after its base letter; keep the base, drop the mark.
                continue;
            }

            Flush(buffer, tokens);
        }

        Flush(buffer, tokens);
        return tokens.ToArray();
    }

    private void Flush(StringBuilder buffer, List<string> tokens)
    {
        if (buffer.Length == 0)
        {
            return;
        }

        string token = buffer.ToString();
        buffer.Clear();
        if (!_stopwords.Contains(token))
        {
            tokens.Add(token);
        }
    }
}
