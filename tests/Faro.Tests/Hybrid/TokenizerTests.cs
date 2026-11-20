using Faro.Hybrid;
using Xunit;

namespace Faro.Tests.Hybrid;

public class TokenizerTests
{
    [Fact]
    public void LowercasesAndSplitsOnPunctuation()
    {
        string[] tokens = Tokenizer.Default.Tokenize("The Quick, brown FOX!");

        Assert.Equal(["quick", "brown", "fox"], tokens);
    }

    [Fact]
    public void FoldsAccents_SoAcacaoAndAcaoTurnIntoTheSameToken()
    {
        var tokenizer = new Tokenizer(Stopwords.None, foldAccents: true);

        Assert.Equal(["acao"], tokenizer.Tokenize("ação"));
        Assert.Equal(["acao"], tokenizer.Tokenize("ACÃO"));
        Assert.Equal(tokenizer.Tokenize("ação"), tokenizer.Tokenize("acao"));
        Assert.Equal(["coracao"], tokenizer.Tokenize("coração"));
    }

    [Fact]
    public void AccentFoldingCanBeDisabled()
    {
        var tokenizer = new Tokenizer(Stopwords.None, foldAccents: false);

        Assert.Equal(["ação"], tokenizer.Tokenize("ação"));
        Assert.NotEqual(tokenizer.Tokenize("ação"), tokenizer.Tokenize("acao"));
    }

    [Fact]
    public void EnglishStopwords_AreRemoved()
    {
        var tokenizer = new Tokenizer(Stopwords.English, foldAccents: false);

        Assert.Equal(["cat", "sat", "mat"], tokenizer.Tokenize("The cat sat on the mat"));
    }

    [Fact]
    public void PortugueseStopwords_AreRemovedInAccentedAndFoldedForms()
    {
        var tokenizer = new Tokenizer(Stopwords.Portuguese, foldAccents: true);

        Assert.Equal(["gato", "aqui"], tokenizer.Tokenize("O gato não está aqui"));
        Assert.Equal(["gato", "aqui"], tokenizer.Tokenize("o gato nao esta aqui"));
    }

    [Fact]
    public void UnicodeNormalization_HandlesDecomposedInput()
    {
        var tokenizer = new Tokenizer(Stopwords.None, foldAccents: true);

        // "a" + combining tilde is canonically equivalent to "ã" before folding.
        Assert.Equal(["a"], tokenizer.Tokenize("a\u0303"));
    }

    [Fact]
    public void NonLatinAndEmoji_AreSplitCleanly()
    {
        var tokenizer = new Tokenizer(Stopwords.None, foldAccents: true);

        Assert.Equal(["привет", "世界", "42"], tokenizer.Tokenize("Привет, 世界 👋 42"));
    }

    [Fact]
    public void EmptyAndOnlyStopwords_ProduceNoTokens()
    {
        var tokenizer = Tokenizer.Default;

        Assert.Empty(tokenizer.Tokenize(""));
        Assert.Empty(tokenizer.Tokenize("the and or"));
    }

    [Fact]
    public void StopwordSet_IsComposableAndPluggable()
    {
        Stopwords custom = Stopwords.FromTokens(["custom"]);
        Stopwords combined = Stopwords.English.Union(custom);
        var tokenizer = new Tokenizer(combined, foldAccents: false);

        Assert.Equal(["keep"], tokenizer.Tokenize("the custom keep"));
        Assert.True(combined.Contains("custom"));
        Assert.False(Stopwords.English.Contains("custom"));
        Assert.False(Stopwords.None.Contains("the"));
    }

    [Fact]
    public void NullArguments_Throw()
    {
        Assert.Throws<ArgumentNullException>(() => new Tokenizer(null!));
        Assert.Throws<ArgumentNullException>(() => Tokenizer.Default.Tokenize(null!));
        Assert.Throws<ArgumentNullException>(() => Stopwords.FromTokens(null!));
    }
}
