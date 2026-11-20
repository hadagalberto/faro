using System.Buffers.Binary;
using Faro.Persistence;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Persistence;

/// <summary>
/// Simulates crashes by truncating or corrupting the WAL at arbitrary byte offsets. The expected
/// state is derived independently from the documented frame layout, not from the recovery code.
/// </summary>
public class CrashSimulationTests : PersistenceTestBase
{
    [Fact]
    public void RandomTruncationOffsets_YieldConsistentPrefix()
    {
        string source = DbPath("source.faro");
        const int UpsertCount = 14;
        using (var db = FaroDb.Open(source))
        {
            FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options());
            for (int i = 0; i < UpsertCount; i++)
            {
                docs.Upsert(i, OneHot(i), MakeDoc(i));
            }

            Assert.True(docs.Delete(9));
            docs.Upsert(99, OneHot(99 % 16), MakeDoc(99));
        }

        byte[] wal = File.ReadAllBytes(source + WalFile.Suffix);
        long[] ends = FrameEnds(wal);
        Assert.Equal(UpsertCount + 3, ends.Length); // create + upserts + delete + final upsert

        // (frame index, key, isDelete). Frame 0 is the collection definition.
        var operations = new List<(int Frame, int Key, bool Delete)>();
        for (int i = 0; i < UpsertCount; i++)
        {
            operations.Add((i + 1, i, false));
        }

        operations.Add((UpsertCount + 1, 9, true));
        operations.Add((UpsertCount + 2, 99, false));

        var offsets = new List<int> { 0, 8, 16, 17, wal.Length - 1 };
        offsets.AddRange(ends.Select(end => checked((int)end)));
        var random = new Random(0x5EED);
        for (int i = 0; i < 48; i++)
        {
            offsets.Add(random.Next(0, wal.Length + 1));
        }

        for (int caseNumber = 0; caseNumber < offsets.Count; caseNumber++)
        {
            int offset = offsets[caseNumber];
            HashSet<int> expected = ExpectedKeys(operations, ends, offset);

            string directory = Path.Combine(Root, $"case-{caseNumber:D3}");
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, "db.faro");
            File.WriteAllBytes(path + WalFile.Suffix, wal.AsSpan(0, offset).ToArray());

            using (var db = FaroDb.Open(path))
            {
                FaroDbCollection<Doc> recovered = db.GetOrCreateCollection("docs", Options());
                Assert.Equal(expected.Count, recovered.Count);
                for (int key = 0; key <= 99; key++)
                {
                    if (key < UpsertCount || key == 99)
                    {
                        Assert.Equal(expected.Contains(key), recovered.ContainsKey(key));
                    }
                }

                foreach (int key in expected)
                {
                    Assert.True(recovered.TryGet(key, out Doc? doc), $"payload missing at offset {offset}");
                    Assert.Equal(MakeDoc(key).Year, doc!.Year);
                    SearchHit<int>[] hit = recovered.Search(OneHot(key % 16), 1);
                    Assert.NotEmpty(hit);
                }
            }

            // Recovery must truncate the torn tail so a second open and subsequent writes are safe.
            using (var db = FaroDb.Open(path))
            {
                FaroDbCollection<Doc> reopened = db.GetOrCreateCollection("docs", Options());
                Assert.Equal(expected.Count, reopened.Count);
                reopened.Upsert(500, OneHot(4), MakeDoc(500));
            }

            using var final = FaroDb.Open(path);
            FaroDbCollection<Doc> finalDocs = final.GetOrCreateCollection("docs", Options());
            Assert.Equal(expected.Count + 1, finalDocs.Count);
        }
    }

    [Fact]
    public void TornTail_IsTruncatedAndNewRecordsSurvive()
    {
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> written = db.GetOrCreateCollection("docs", Options());
            for (int i = 0; i < 3; i++)
            {
                written.Upsert(i, OneHot(i), MakeDoc(i));
            }
        }

        string walPath = path + WalFile.Suffix;
        long committedLength = new FileInfo(walPath).Length;
        using (var stream = new FileStream(walPath, FileMode.Append, FileAccess.Write))
        {
            stream.Write([0x31, 0x32, 0x33, 0x34, 0x35]);
        }

        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> recovered = db.GetOrCreateCollection("docs", Options());
            Assert.Equal(3, recovered.Count);
            recovered.Upsert(3, OneHot(3), MakeDoc(3));
        }

        Assert.True(new FileInfo(walPath).Length > committedLength);
        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> finalDocs = reopened.GetOrCreateCollection("docs", Options());
        Assert.Equal(4, finalDocs.Count);
        Assert.True(finalDocs.ContainsKey(3));
    }

    [Fact]
    public void CorruptRecordChecksum_DropsThatRecordAndEverythingAfterIt()
    {
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> written = db.GetOrCreateCollection("docs", Options());
            for (int i = 0; i < 5; i++)
            {
                written.Upsert(i, OneHot(i), MakeDoc(i));
            }
        }

        string walPath = path + WalFile.Suffix;
        byte[] wal = File.ReadAllBytes(walPath);
        long[] ends = FrameEnds(wal);
        int thirdRecordStart = checked((int)ends[2]); // create + upserts 0,1 are complete
        wal[thirdRecordStart + 12] ^= 0xFF; // flip a byte inside the third upsert's payload
        File.WriteAllBytes(walPath, wal);

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> recovered = reopened.GetOrCreateCollection("docs", Options());
        Assert.Equal(2, recovered.Count);
        Assert.True(recovered.ContainsKey(0) && recovered.ContainsKey(1));
        Assert.False(recovered.ContainsKey(2));
    }

    private static HashSet<int> ExpectedKeys(
        IReadOnlyList<(int Frame, int Key, bool Delete)> operations,
        IReadOnlyList<long> frameEnds,
        int offset)
    {
        var keys = new HashSet<int>();
        foreach ((int frame, int key, bool delete) in operations)
        {
            if (frameEnds[frame] > offset)
            {
                continue;
            }

            if (delete)
            {
                keys.Remove(key);
            }
            else
            {
                keys.Add(key);
            }
        }

        return keys;
    }

    /// <summary>Independently parses the documented WAL framing: 4-byte length, 4-byte CRC, 1-byte type, payload.</summary>
    private static long[] FrameEnds(byte[] wal)
    {
        Assert.Equal("FAROWAL1"u8.ToArray(), wal.AsSpan(0, 8).ToArray());
        Assert.Equal(1u, BinaryPrimitives.ReadUInt32LittleEndian(wal.AsSpan(8, 4)));

        var ends = new List<long>();
        int offset = 16;
        while (offset < wal.Length)
        {
            int length = BinaryPrimitives.ReadInt32LittleEndian(wal.AsSpan(offset, 4));
            Assert.True(length > 0);
            offset += 9 + length;
            Assert.True(offset <= wal.Length);
            ends.Add(offset);
        }

        return ends.ToArray();
    }
}
