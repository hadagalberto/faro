namespace Faro.Hybrid;

/// <summary>
/// An immutable set of tokens ignored by the BM25 tokenizer.
/// </summary>
public sealed class Stopwords
{
    private static readonly string[] EnglishWords =
    [
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me",
        "more", "most", "my", "myself", "no", "nor", "not", "of", "on", "once", "only", "or", "other",
        "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some",
        "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
        "with", "you", "your", "yours", "yourself", "yourselves",
    ];

    private static readonly string[] PortugueseWords =
    [
        "a", "ao", "aos", "as", "até", "ate", "com", "como", "da", "das", "de", "do", "dos", "e", "ela",
        "elas", "ele", "eles", "em", "entre", "era", "eras", "é", "e", "essa", "essas", "esse", "esses",
        "esta", "estas", "este", "estes", "eu", "foi", "foram", "há", "ha", "isso", "isto", "já", "ja", "lhe",
        "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "nas",
        "não", "nao", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou",
        "para", "por", "qual", "quando", "que", "quem", "se", "sem", "ser", "seu", "seus", "sua",
        "suas", "são", "sao", "também", "tambem", "te", "tem", "tenho", "ter", "tu", "um", "uma", "umas",
        "uns", "vezes", "você", "voce", "vocês", "voces", "vos",
    ];

    private readonly HashSet<string> _words;

    private Stopwords(HashSet<string> words)
    {
        _words = words;
    }

    /// <summary>No stopwords: every token is indexed.</summary>
    public static Stopwords None { get; } = new(new HashSet<string>(StringComparer.Ordinal));

    /// <summary>English stopwords.</summary>
    public static Stopwords English { get; } = FromTokens(EnglishWords);

    /// <summary>Portuguese stopwords, including common accented forms.</summary>
    public static Stopwords Portuguese { get; } = FromTokens(PortugueseWords);

    /// <summary>English and Portuguese stopwords combined (the tokenizer default).</summary>
    public static Stopwords EnglishAndPortuguese { get; } = English.Union(Portuguese);

    /// <summary>Creates a stopword set from caller-supplied tokens.</summary>
    /// <param name="tokens">Tokens to ignore; comparison is ordinal.</param>
    /// <returns>The stopword set.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="tokens"/> is <see langword="null"/>.</exception>
    public static Stopwords FromTokens(IEnumerable<string> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        return new Stopwords(new HashSet<string>(tokens, StringComparer.Ordinal));
    }

    /// <summary>Creates a new set containing the union of this set and another.</summary>
    /// <param name="other">The other stopword set.</param>
    /// <returns>The combined set.</returns>
    public Stopwords Union(Stopwords other)
    {
        ArgumentNullException.ThrowIfNull(other);
        var combined = new HashSet<string>(_words, StringComparer.Ordinal);
        combined.UnionWith(other._words);
        return new Stopwords(combined);
    }

    /// <summary>Returns <see langword="true"/> when <paramref name="token"/> should be ignored.</summary>
    /// <param name="token">An already-normalized token.</param>
    public bool Contains(string token) => _words.Contains(token);
}
