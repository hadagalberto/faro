using System.Buffers.Binary;
using System.Diagnostics;
using Faro.Persistence;
using Faro.Tests.Shared;
using Xunit;

namespace Faro.Tests.Persistence;

/// <summary>
/// Batched durability used to leave records inside the 64 KB FileStream buffer; a FailFast crash
/// lost them. Also covers a corrupt WAL header wiping the whole log and structurally invalid
/// records being skipped instead of stopping replay at the last valid prefix.
/// </summary>
public class WalDurabilityTests : PersistenceTestBase
{
    [Fact]
    public void BatchedBatch_SurvivesFailFastCrash()
    {
        const int Count = 520;
        string path = DbPath("crash.faro");
        string project = Path.Combine(RepositoryRoot(), "tests", "Faro.CrashWorker", "Faro.CrashWorker.csproj");

        var start = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        start.ArgumentList.Add("run");
        start.ArgumentList.Add("--project");
        start.ArgumentList.Add(project);
        start.ArgumentList.Add("-c");
        start.ArgumentList.Add("Release");
        start.ArgumentList.Add("-f");
#if NET10_0
        start.ArgumentList.Add("net10.0");
#else
        start.ArgumentList.Add("net8.0");
#endif
        start.ArgumentList.Add("--");
        start.ArgumentList.Add(path);
        start.ArgumentList.Add(((int)Durability.Batched).ToString());
        start.ArgumentList.Add(Count.ToString());

        using Process? child = Process.Start(start);
        Assert.NotNull(child);
        string stdout = child.StandardOutput.ReadToEnd();
        string stderr = child.StandardError.ReadToEnd();
        Assert.True(child.WaitForExit(TimeSpan.FromMinutes(3)), "crash worker timed out");
        Assert.NotEqual(0, child.ExitCode); // FailFast must terminate the process abnormally
        // A build failure also exits non-zero; require a WAL with real frames so that cannot pass.
        Assert.True(File.Exists(path + WalFile.Suffix) &&
                    new FileInfo(path + WalFile.Suffix).Length > 16,
            "crash worker produced no WAL records: " + stderr + stdout);

        // Hand-derived: every record of a completed batch was fsynced before the crash.
        using var db = FaroDb.Open(path);
        FaroDbCollection<Doc> loaded = db.GetOrCreateCollection("docs", Options());
        Assert.Equal(Count, loaded.Count);
        for (int i = 0; i < Count; i++)
        {
            Assert.True(loaded.ContainsKey(i), $"key {i} lost across FailFast crash");
        }
    }

    [Fact]
    public void CorruptWalHeader_ThrowsByDefaultAndSalvageKeepsOriginalBytes()
    {
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            db.GetOrCreateCollection("docs", Options()).Upsert(1, OneHot(1), MakeDoc(1));
        }

        string walPath = path + WalFile.Suffix;
        byte[] wal = File.ReadAllBytes(walPath);
        byte[] corrupt = (byte[])wal.Clone();
        corrupt[0] ^= 0xFF; // break the WAL magic; header CRC now fails
        File.WriteAllBytes(walPath, corrupt);

        Assert.Throws<FaroCorruptionException>(() =>
        {
            using var _ = FaroDb.Open(path);
        });
        Assert.Equal(corrupt, File.ReadAllBytes(walPath)); // default open never destroys the log

        using var salvaged = FaroDb.OpenWithRecovery(path, null, out CorruptionReport report);
        Assert.Contains(report.Issues, issue => issue.Reason.Contains("WAL header", StringComparison.OrdinalIgnoreCase));
        Assert.True(File.Exists(walPath + ".corrupt"));
        Assert.Equal(corrupt, File.ReadAllBytes(walPath + ".corrupt"));
        Assert.Equal(0, salvaged.GetOrCreateCollection("docs", Options()).Count);
    }

    [Fact]
    public void StructurallyInvalidRecord_StopsReplayAtLastValidPrefix()
    {
        string path = DbPath();
        using (var db = FaroDb.Open(path))
        {
            FaroDbCollection<Doc> docs = db.GetOrCreateCollection("docs", Options());
            for (int i = 0; i < 4; i++)
            {
                docs.Upsert(i, OneHot(i), MakeDoc(i));
            }
        }

        string walPath = path + WalFile.Suffix;
        byte[] wal = File.ReadAllBytes(walPath);
        (int Start, int Length)[] frames = WalFrames(wal);
        Assert.Equal(5, frames.Length); // create + 4 upserts

        (int start, int length) = frames[2]; // the upsert for key 1
        int payloadStart = start + 9;
        // Payload layout: collectionId(4), id(4), level(4), deleted(1), vector blob length(4) ...
        const int VectorLengthOffset = 4 + 4 + 4 + 1;
        BinaryPrimitives.WriteInt32LittleEndian(wal.AsSpan(payloadStart + VectorLengthOffset, 4), -1);
        byte type = wal[start + 8];
        ReadOnlySpan<byte> payload = wal.AsSpan(payloadStart, length);
        BinaryPrimitives.WriteUInt32LittleEndian(
            wal.AsSpan(start + 4, 4), Crc32C.Compute([type], payload)); // keep the frame CRC valid
        File.WriteAllBytes(walPath, wal);

        using var reopened = FaroDb.Open(path);
        FaroDbCollection<Doc> recovered = reopened.GetOrCreateCollection("docs", Options());
        // Hand-derived: frames 0 and 1 are valid (create + key 0). Frame 2 is malformed and
        // frames 3 and 4 must NOT be applied even though their CRCs are fine.
        Assert.Equal(1, recovered.Count);
        Assert.True(recovered.ContainsKey(0));
        Assert.False(recovered.ContainsKey(1));
        Assert.False(recovered.ContainsKey(2));
        Assert.False(recovered.ContainsKey(3));
    }

    private static (int Start, int Length)[] WalFrames(byte[] wal)
    {
        Assert.Equal("FAROWAL1"u8.ToArray(), wal.AsSpan(0, 8).ToArray());
        var result = new List<(int, int)>();
        int offset = 16;
        while (offset < wal.Length)
        {
            int length = BinaryPrimitives.ReadInt32LittleEndian(wal.AsSpan(offset, 4));
            Assert.True(length > 0);
            result.Add((offset, length));
            offset += 9 + length;
        }

        return result.ToArray();
    }

    private static string RepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Faro.sln")))
        {
            dir = dir.Parent;
        }

        Assert.NotNull(dir);
        return dir.FullName;
    }
}
