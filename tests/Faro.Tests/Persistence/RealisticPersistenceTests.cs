using Faro.Persistence;
using Faro.Testing;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Persistence;

/// <summary>
/// Persistence tests with realistic (not one-hot) dimensionality and corruption fuzzing. Expected
/// vectors come from a hand-seeded splitmix64 generator in the test, so an exact match proves the
/// file round-tripped the bytes rather than re-deriving them through Faro.
/// </summary>
public class RealisticPersistenceTests : PersistenceTestBase
{
    [Fact]
    public void RandomizedHighDimension_CheckpointRoundTrip_IsBitExact()
    {
        const int Dimensions = 128;
        const int Records = 200;
        const ulong Seed = 0xC0FFEEul;
        string path = DbPath();

        var expected = new float[Records][];
        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options(Dimensions));
            for (int i = 0; i < Records; i++)
            {
                expected[i] = DeterministicRandom.UniformVector(Dimensions, DeterministicRandom.DeriveSeed(Seed, i));
                docs.Upsert(i, expected[i], MakeDoc(i));
            }

            db.Checkpoint();
        }

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> loaded = reopened.GetOrCreateCollection("docs", Options(Dimensions));
        Assert.Equal(Records, loaded.Count);
        var actual = new float[Dimensions];
        for (int i = 0; i < Records; i++)
        {
            Assert.True(loaded.Index.TryGetVector(i, actual), $"vector {i} missing after reopen");
            Assert.Equal(expected[i], actual); // bit-exact, not tolerance-based
            Assert.True(loaded.TryGet(i, out Doc? payload));
            Assert.Equal(MakeDoc(i).Year, payload!.Year);
            Assert.Equal(MakeDoc(i).Tenant, payload.Tenant);
        }
    }

    [Fact]
    public void ByteFlipsAndTruncations_NeverYieldSilentlyDifferentData()
    {
        const int Dimensions = 64;
        const int Records = 60;
        const ulong Seed = 0xBADF00Dul;
        string source = DbPath("source.faro");

        var expected = new float[Records][];
        using (var db = FaroDb.Open(source))
        {
            FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options(Dimensions));
            for (int i = 0; i < Records; i++)
            {
                expected[i] = DeterministicRandom.UniformVector(Dimensions, DeterministicRandom.DeriveSeed(Seed, i));
                docs.Upsert(i, expected[i], MakeDoc(i));
            }

            db.Checkpoint();
        }

        byte[] good = File.ReadAllBytes(source);
        CheckpointPage[] pages = CheckpointLayout.Pages(good);
        var mutations = new List<(string Name, Action<byte[]> Mutate)>();

        // One flipped byte in every page kind, at a deterministic interior position.
        foreach (byte kind in new byte[] { 1, 2, 3, 4, 5 })
        {
            CheckpointPage page = pages.First(p => p.Kind == kind);
            long target = page.Offset + page.Length / 2;
            mutations.Add(($"flip-kind-{kind}", bytes => bytes[checked((int)target)] ^= 0x5A));
        }

        // Torn writes: truncate around every page boundary and inside the header.
        foreach (int length in new[] { 0, 8, 64, 127, (int)pages[0].Offset + 1, (int)good.Length - 1 })
        {
            mutations.Add(($"truncate-{length}", bytes => Array.Resize(ref bytes, Math.Min(length, bytes.Length))));
        }

        for (int caseNumber = 0; caseNumber < mutations.Count; caseNumber++)
        {
            (string name, Action<byte[]> mutate) = mutations[caseNumber];
            byte[] damaged = (byte[])good.Clone();
            mutate(damaged);

            string path = DbPath($"fuzz-{caseNumber:D2}-{name}.faro");
            File.WriteAllBytes(path, damaged);
            try
            {
                using var db = FaroDb.Open(path);
                FaroDbCollection<Doc> loaded = db.GetOrCreateCollection("docs", Options(Dimensions));
                // A successful open must contain the exact original data — never a partial or mutated DB.
                Assert.Equal(Records, loaded.Count);
                var actual = new float[Dimensions];
                for (int i = 0; i < Records; i++)
                {
                    Assert.True(loaded.Index.TryGetVector(i, actual));
                    Assert.Equal(expected[i], actual);
                }
            }
            catch (IOException)
            {
                // FaroCorruptionException derives from IOException: loud failure is the other
                // allowed outcome. Silent data loss is not.
            }
        }
    }
}
