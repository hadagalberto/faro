using System.Buffers.Binary;

namespace Faro.Persistence;

/// <summary>One decoded WAL record. The payload is parsed by <see cref="FaroDb"/> replay.</summary>
internal readonly record struct WalRecord(byte Type, ReadOnlyMemory<byte> Payload);

/// <summary>Result of scanning a write-ahead log, including any torn tail.</summary>
internal sealed class WalReplay
{
    internal WalReplay(List<WalRecord> records, long validLength, bool hadTornTail)
    {
        Records = records;
        ValidLength = validLength;
        HadTornTail = hadTornTail;
    }

    internal List<WalRecord> Records { get; }
    internal long ValidLength { get; }
    internal bool HadTornTail { get; }
}

/// <summary>
/// Append-only write-ahead log with a CRC32C per record (Castagnoli). A partial or corrupt
/// trailing record is never replayed; when truncation is enabled the file is shortened to the end
/// of the last valid record so subsequent appends remain well formed.
/// </summary>
internal sealed class WalFile : IDisposable
{
    internal const string Suffix = ".wal";
    private const int HeaderSize = 16;
    private const int MaxRecordBytes = 256 * 1024 * 1024;
    private static readonly byte[] Magic = "FAROWAL1"u8.ToArray();

    private readonly string _path;
    private readonly Durability _durability;
    private readonly object _gate = new();
    private FileStream? _stream;
    private bool _dirty;
    private bool _disposed;

    internal WalFile(string path, Durability durability, bool truncateTornTail)
    {
        _path = path;
        _durability = durability;
        Replay = Scan(path, truncateTornTail);
        _stream = new FileStream(
            path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read,
            bufferSize: 64 * 1024, FileOptions.SequentialScan);
        _stream.Seek(0, SeekOrigin.End);

        if (_stream.Length < HeaderSize)
        {
            WriteHeader();
        }
    }

    internal WalReplay Replay { get; }

    internal string Path => _path;

    /// <summary>Appends one record. The caller builds the payload with a <see cref="ByteWriter"/>.</summary>
    internal void Append(byte type, ByteWriter payload)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(payload);

        ReadOnlySpan<byte> bytes = payload.Span;
        if (bytes.Length > MaxRecordBytes)
        {
            throw new InvalidOperationException("WAL record exceeds the maximum supported size");
        }

        var frame = new byte[9 + bytes.Length];
        BinaryPrimitives.WriteInt32LittleEndian(frame.AsSpan(0, 4), bytes.Length);
        Span<byte> typeByte = stackalloc byte[1];
        typeByte[0] = type;
        BinaryPrimitives.WriteUInt32LittleEndian(frame.AsSpan(4, 4), Crc32C.Compute(typeByte, bytes));
        frame[8] = type;
        bytes.CopyTo(frame.AsSpan(9));

        lock (_gate)
        {
            _stream!.Write(frame);
            _dirty = true;
            if (_durability == Durability.Always)
            {
                _stream.Flush(flushToDisk: true);
                _dirty = false;
            }
        }
    }

    /// <summary>Flushes pending records according to the configured durability mode.</summary>
    internal void Flush()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            if (_durability != Durability.None)
            {
                _stream!.Flush(flushToDisk: true);
            }

            _dirty = false;
        }
    }

    /// <summary>Removes all WAL records. Called only after a checkpoint file is durable.</summary>
    internal void ResetAfterCheckpoint()
    {
        lock (_gate)
        {
            _stream!.SetLength(0);
            _stream.Seek(0, SeekOrigin.Begin);
            WriteHeader();
            _stream.Flush(flushToDisk: true);
            _dirty = false;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        lock (_gate)
        {
            if (_dirty && _durability == Durability.Batched)
            {
                _stream!.Flush(flushToDisk: true);
            }

            _dirty = false;
            _stream!.Dispose();
            _stream = null;
            _disposed = true;
        }
    }

    private void WriteHeader()
    {
        var header = new byte[HeaderSize];
        Magic.CopyTo(header, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8, 4), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(12, 4), Crc32C.Compute(header.AsSpan(0, 12)));
        _stream!.Write(header);
    }

    /// <summary>Scans a WAL file and returns the longest prefix of well-formed records.</summary>
    internal static WalReplay Scan(string path, bool truncateTornTail)
    {
        if (!File.Exists(path) || new FileInfo(path).Length == 0)
        {
            return new WalReplay([], 0, false);
        }

        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length < HeaderSize ||
            !bytes.AsSpan(0, Magic.Length).SequenceEqual(Magic) ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(8, 4)) != 1 ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(12, 4)) != Crc32C.Compute(bytes.AsSpan(0, 12)))
        {
            if (truncateTornTail)
            {
                File.WriteAllBytes(path, []);
            }

            return new WalReplay([], 0, true);
        }

        var records = new List<WalRecord>();
        int offset = HeaderSize;
        bool torn = false;
        while (offset < bytes.Length)
        {
            if (bytes.Length - offset < 8)
            {
                torn = true;
                break;
            }

            int length = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset, 4));
            uint checksum = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset + 4, 4));
            if (length <= 0 || length > MaxRecordBytes || length > bytes.Length - offset - 9)
            {
                torn = true;
                break;
            }

            byte type = bytes[offset + 8];
            ReadOnlySpan<byte> payload = bytes.AsSpan(offset + 9, length);
            if (Crc32C.Compute(new[] { type }, payload) != checksum)
            {
                torn = true;
                break;
            }

            records.Add(new WalRecord(type, payload.ToArray()));
            offset += 9 + length;
        }

        if (torn && truncateTornTail && offset != bytes.Length)
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.Read);
            stream.SetLength(offset);
            stream.Flush(flushToDisk: true);
        }

        return new WalReplay(records, offset, torn);
    }
}
