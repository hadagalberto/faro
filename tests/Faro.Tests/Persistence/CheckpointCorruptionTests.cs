using System.Buffers.Binary;
using Faro.Persistence;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Persistence;

/// <summary>
/// The original defect: a page with a bad CRC (or an out-of-range page-table offset) was treated
/// as if it were absent, so a single flipped byte opened an apparently empty database and the next
/// checkpoint made the loss permanent. Expected values below are derived from the file format and
/// the number of records written, never from the recovery code.
/// </summary>
public class CheckpointCorruptionTests : PersistenceTestBase
{
    [Fact]
    public void CorruptVectorPage_DefaultOpen_ThrowsAndNeverMutatesTheFile()
    {
        const int Records = 1000;
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options());
            for (int i = 0; i < Records; i++)
            {
                docs.Upsert(i, OneHot(i % 16), MakeDoc(i));
            }

            db.Checkpoint();
        }

        byte[] original = File.ReadAllBytes(path);
        CheckpointPage vectorPage = CheckpointLayout.Pages(original).Single(p => p.Kind == 2);
        byte[] corrupt = (byte[])original.Clone();
        corrupt[checked((int)(vectorPage.Offset + vectorPage.Length / 2))] ^= 0x01;
        File.WriteAllBytes(path, corrupt);

        // Hand-derived: kind 2 is the vector page, so corruption must be reported as vector damage.
        FaroCorruptionException error = Assert.Throws<FaroCorruptionException>(() =>
        {
            using var _ = FaroDb.Open(path);
        });
        Assert.Contains("vector", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(corrupt, File.ReadAllBytes(path)); // a failed open must not rewrite the file

        using var salvaged = FaroDb.OpenWithRecovery(path, null, out CorruptionReport report);
        Assert.True(report.HasIssues);
        Assert.Contains(report.Issues, issue => issue.PageKind == 2 && issue.Reason.Contains("checksum", StringComparison.OrdinalIgnoreCase));
        // Hand-derived: a collection without a valid vector page cannot yield any nodes.
        Assert.Equal(0, salvaged.GetOrCreateCollection("docs", Options()).Count);
    }

    [Fact]
    public void OutOfBoundsPageOffset_ThrowsCorruptionException()
    {
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options());
            for (int i = 0; i < 32; i++)
            {
                docs.Upsert(i, OneHot(i % 16), MakeDoc(i));
            }

            db.Checkpoint();
        }

        byte[] file = File.ReadAllBytes(path);
        CheckpointPage[] pages = CheckpointLayout.Pages(file);
        int vectorIndex = Array.FindIndex(pages, p => p.Kind == 2);
        Assert.True(vectorIndex >= 0);

        long tableOffset = BinaryPrimitives.ReadInt64LittleEndian(file.AsSpan(32, 8));
        int entry = checked((int)tableOffset) + vectorIndex * CheckpointLayout.PageRecordSize;
        // Hand-derived: an offset one byte past EOF can never contain a valid page.
        BinaryPrimitives.WriteInt64LittleEndian(file.AsSpan(entry + 8, 8), file.Length + 1L);
        File.WriteAllBytes(path, file);

        Assert.Throws<FaroCorruptionException>(() =>
        {
            using var _ = FaroDb.Open(path);
        });
    }

    [Fact]
    public void CorruptHeader_StillReportsInvalidData()
    {
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            db.GetOrCreateCollection("docs", Options()).Upsert(1, OneHot(1), MakeDoc(1));
            db.Checkpoint();
        }

        byte[] file = File.ReadAllBytes(path);
        file[64] ^= 0xFF; // breaks the header checksum
        File.WriteAllBytes(path, file);

        // Hand-derived: the header covers bytes [0,64); a 64-byte CRC failure is not a valid file.
        Assert.Throws<InvalidDataException>(() =>
        {
            using var _ = FaroDb.Open(path);
        });
    }
}
