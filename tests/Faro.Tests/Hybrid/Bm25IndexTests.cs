using Faro.Hybrid;
using Xunit;

namespace Faro.Tests.Hybrid;

public class Bm25IndexTests
{
    private static readonly Tokenizer Plain = new(Stopwords.None, foldAccents: false);

    private static Bm25Index Corpus()
    {
        var index = new Bm25Index(); // k1 = 1.5, b = 0.75
        index.AddOrReplace(0, "the cat sat", Plain);
        index.AddOrReplace(1, "the dog sat", Plain);
        index.AddOrReplace(2, "cat cat dog", Plain);
        return index;
    }

    [Fact]
    public void AddOrReplace_BuildsPostingsAndLengths()
    {
        Bm25Index index = Corpus();

        Assert.Equal(3, index.DocumentCount);
        Assert.Equal(4, index.VocabularyCount); // the, cat, sat, dog
        Assert.Equal(3, index.DocumentLength(0));
        Assert.Equal(2, index.TermFrequency(2, "cat"));
        Assert.Equal(1, index.TermFrequency(0, "cat"));
        Assert.Equal(0, index.TermFrequency(0, "missing"));
        Assert.Equal(0, index.DocumentLength(999));
    }

    [Fact]
    public void HandComputedScore_SingleTermTfOne()
    {
        Bm25Index index = Corpus();

        // df(cat)=2, N=3, avgdl=3, k1=1.5, b=0.75:
        // idf = ln(1 + (3-2+0.5)/(2+0.5)) = ln(1.6) = 0.47000362924573563
        // tf' = 1*2.5 / (1 + 1.5*(1-0.75+0.75*1)) = 1
        Assert.Equal(0.47000362924573563, index.Score("cat", Plain, 0), 12);
        Assert.Equal(0.47000362924573563, index.Score("sat", Plain, 1), 12);
    }

    [Fact]
    public void HandComputedScore_TermFrequencySaturation()
    {
        Bm25Index index = Corpus();

        // tf(cat)=2 in doc 2: tf' = 2*2.5/(2+1.5) = 1.4285714285714286
        Assert.Equal(0.67143375606533662, index.Score("cat", Plain, 2), 12);
        Assert.True(index.Score("cat", Plain, 2) > index.Score("cat", Plain, 0));
    }

    [Fact]
    public void HandComputedScore_MultipleQueryTermsAndCommonTerm()
    {
        Bm25Index index = Corpus();

        // "cat sat" in doc 0 is the sum of two identical idf contributions.
        Assert.Equal(0.94000725849147126, index.Score("cat sat", Plain, 0), 12);

        // df(the)=2 (docs 0 and 1): the same idf as "cat" and "sat".
        Assert.Equal(0.47000362924573563, index.Score("the", Plain, 0), 12);
    }

    [Fact]
    public void Score_ZeroForMissingTermDocumentOrEmptyQuery()
    {
        Bm25Index index = Corpus();

        Assert.Equal(0, index.Score("cat", Plain, 1));
        Assert.Equal(0, index.Score("cat", Plain, 999));
        Assert.Equal(0, index.Score("", Plain, 0));
        Assert.Equal(0, new Bm25Index().Score("cat", Plain, 0));
    }

    [Fact]
    public void Search_RanksByScoreAndCanBeRestricted()
    {
        Bm25Index index = Corpus();

        (int DocId, double Score)[] hits = index.Search("cat", Plain, limit: 5);

        Assert.Equal([2, 0], hits.Select(hit => hit.DocId).ToArray());
        Assert.Equal(0.67143375606533662, hits[0].Score, 12);
        Assert.Equal(0.47000362924573563, hits[1].Score, 12);

        (int DocId, double Score)[] filtered = index.Search("cat", Plain, limit: 5, include: id => id == 0);
        Assert.Equal([0], filtered.Select(hit => hit.DocId).ToArray());

        Assert.Empty(index.Search("the and or", new Tokenizer(Stopwords.English), 5));
    }

    [Fact]
    public void Replace_RecomputesDocumentFrequencyAndLength()
    {
        Bm25Index index = Corpus();

        index.AddOrReplace(2, "dog", Plain);

        Assert.Equal(0, index.TermFrequency(2, "cat"));
        Assert.Equal(1, index.TermFrequency(2, "dog"));
        Assert.Equal(1, index.DocumentLength(2));

        // df(cat)=1, avgdl=7/3, dl(doc0)=3: idf=ln(8/3), tf'=2.5/2.8214285714285712
        Assert.Equal(0.86908921152937779, index.Score("cat", Plain, 0), 12);
    }

    [Fact]
    public void Remove_DropsPostingsAndDocument()
    {
        Bm25Index index = Corpus();

        Assert.True(index.Remove(0));
        Assert.False(index.Remove(0));
        Assert.Equal(2, index.DocumentCount);
        Assert.Equal(0, index.Score("cat", Plain, 0));
        Assert.Equal(4, index.VocabularyCount); // doc1 still has "the", doc2 still has "cat"
    }

    [Fact]
    public void AccentFoldedQuery_MatchesAccentedDocument()
    {
        var index = new Bm25Index();
        var folder = new Tokenizer(Stopwords.None, foldAccents: true);
        index.AddOrReplace(0, "ação coração avião", folder);

        Assert.True(index.Score("acao", folder, 0) > 0);
        Assert.True(index.Score("coracao", folder, 0) > 0);
        Assert.All(index.Search("acao coracao", folder, 1), hit => Assert.Equal(0, hit.DocId));
    }

    [Fact]
    public void InvalidArguments_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Bm25Index(0, 0.75));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Bm25Index(1.5, -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Bm25Index(1.5, 1.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Corpus().AddOrReplace(-1, "x", Plain));
        Assert.Throws<ArgumentNullException>(() => Corpus().AddOrReplace(0, null!, Plain));
        Assert.Throws<ArgumentNullException>(() => Corpus().AddOrReplace(0, "x", null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => Corpus().Search("cat", Plain, 0));
        Assert.Contains("k1=1.5", Corpus().ToString());
    }
}
