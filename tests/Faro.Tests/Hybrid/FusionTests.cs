using Faro.Hybrid;
using Xunit;

namespace Faro.Tests.Hybrid;

public class FusionTests
{
    private static SearchResult[] VectorHits() =>
    [
        new(10, 1f),
        new(20, 0.5f),
        new(30, 0f),
    ];

    private static (int DocId, double Score)[] TextHits() =>
    [
        (20, 10.0),
        (10, 5.0),
        (40, 0.0),
    ];

    [Fact]
    public void Rrf_MatchesHandComputedScoresAndRanks()
    {
        HybridResult[] fused = Fusion.Rrf(60).Fuse(VectorHits(), TextHits(), lowerIsBetterVectorScore: false, limit: 10);

        Assert.Equal([10, 20, 30, 40], fused.Select(hit => hit.Id).ToArray());

        // rank 1 + rank 2: 1/61 + 1/62; rank 3 alone: 1/63
        const double top = 1.0 / 61 + 1.0 / 62;
        Assert.Equal(top, fused[0].Score, 7);
        Assert.Equal(top, fused[1].Score, 7);
        Assert.Equal(1.0 / 63, fused[2].Score, 7);
        Assert.Equal(1.0 / 63, fused[3].Score, 7);

        Assert.Equal(1, fused[0].VectorRank);
        Assert.Equal(2, fused[0].TextRank);
        Assert.Equal(2, fused[1].VectorRank);
        Assert.Equal(1, fused[1].TextRank);
        Assert.Equal(0, fused[3].VectorRank);
        Assert.Equal(3, fused[3].TextRank);
        Assert.Equal(1f, fused[0].VectorScore!.Value);
        Assert.Equal(5.0, fused[0].TextScore!.Value);
    }

    [Fact]
    public void Rrf_KConstantChangesContribution()
    {
        HybridResult[] small = Fusion.Rrf(1).Fuse(VectorHits(), TextHits(), false, 10);
        HybridResult[] large = Fusion.Rrf(100).Fuse(VectorHits(), TextHits(), false, 10);

        Assert.True(small[0].Score > large[0].Score);
    }

    [Fact]
    public void Weighted_MatchesHandComputedScores()
    {
        HybridResult[] fused = Fusion.Weighted(0.5, 0.5)
            .Fuse(VectorHits(), TextHits(), lowerIsBetterVectorScore: false, limit: 10);

        Assert.Equal([10, 20, 30, 40], fused.Select(hit => hit.Id).ToArray());
        // 10: 0.5*1 + 0.5*0.5; 20: 0.5*0.5 + 0.5*1
        Assert.Equal(0.75f, fused[0].Score, 7);
        Assert.Equal(0.75f, fused[1].Score, 7);
        // id 30 is last in the vector list and id 40 is last in the text list: both normalize to 0.
        Assert.Equal(0f, fused[2].Score, 7);
        Assert.Equal(0f, fused[3].Score, 7);
    }

    [Fact]
    public void Weighted_HigherWeightsChangeOrdering()
    {
        HybridResult[] textHeavy = Fusion.Weighted(0.1, 0.9)
            .Fuse(VectorHits(), TextHits(), lowerIsBetterVectorScore: false, limit: 10);

        Assert.Equal(20, textHeavy[0].Id);
    }

    [Fact]
    public void Weighted_LowerIsBetterVectorScoresAreInverted()
    {
        SearchResult[] distances = [new(1, 0.1f), new(2, 0.5f), new(3, 1.0f)];
        HybridResult[] fused = Fusion.Weighted(1, 0)
            .Fuse(distances, [], lowerIsBetterVectorScore: true, limit: 10);

        Assert.Equal([1, 2, 3], fused.Select(hit => hit.Id).ToArray());
        Assert.Equal(1f, fused[0].Score, 7);
        Assert.Equal((1f - 0.5f) / 0.9f, fused[1].Score, 7);
        Assert.Equal(0f, fused[2].Score, 7);
    }

    [Fact]
    public void SingleItemList_NormalizesToBest()
    {
        HybridResult[] fused = Fusion.Weighted(1, 1)
            .Fuse([new SearchResult(7, 0.2f)], [(7, 3.0)], lowerIsBetterVectorScore: true, limit: 10);

        Assert.Single(fused);
        Assert.Equal(2f, fused[0].Score, 7);
    }

    [Fact]
    public void EmptyLists_ProduceNoResults()
    {
        Assert.Empty(Fusion.Rrf().Fuse([], [], false, 10));
        Assert.Empty(Fusion.Weighted().Fuse(VectorHits(), TextHits(), false, 0));
    }

    [Fact]
    public void InvalidFusionArguments_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Fusion.Rrf(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fusion.Weighted(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fusion.Weighted(0, 0));
    }
}
