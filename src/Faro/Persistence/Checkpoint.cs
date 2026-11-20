using System.Buffers.Binary;
using Faro.Hybrid;

namespace Faro.Persistence;

/// <summary>Everything read from a valid checkpoint file.</summary>
internal sealed class CheckpointData
{
    internal CheckpointData(
        ulong databaseId,
        Dictionary<int, CollectionDefinition> definitions,
        Dictionary<int, PersistedNode[]> nodes,
        Dictionary<int, PersistedRecord[]> records)
    {
        DatabaseId = databaseId;
        Definitions = definitions;
        Nodes = nodes;
        Records = records;
    }

    internal ulong DatabaseId { get; }
    internal Dictionary<int, CollectionDefinition> Definitions { get; }
    internal Dictionary<int, PersistedNode[]> Nodes { get; }
    internal Dictionary<int, PersistedRecord[]> Records { get; }
}

/// <summary>
/// Reads and writes the versioned single-file checkpoint described in <c>docs/file-format.md</c>.
/// </summary>
/// <remarks>
/// A checkpoint is always written to a temporary file, flushed, then atomically renamed over the
/// main file. The header is filled last so a partially written file is never mistaken for a
/// committed checkpoint. Every page carries its own CRC32C; a page with a bad checksum is ignored.
/// </remarks>
internal static class Checkpoint
{
    private const int HeaderSize = 128;
    private const int PageRecordSize = 32;
    private const int Alignment = 64;
    private const uint Version = 1;
    private const byte PageCollectionDef = 1;
    private const byte PageVector = 2;
    private const byte PageGraph = 3;
    private const byte PagePayload = 4;
    private const byte PageText = 5;
    private static readonly byte[] Magic = "FARO"u8.ToArray();

    /// <summary>Writes a new checkpoint file and atomically installs it.</summary>
    internal static void Write(
        string path,
        ulong databaseId,
        CollectionState[] collections,
        IReadOnlyDictionary<int, Bm25Snapshot> textSnapshots)
    {
        var pages = new List<(byte Kind, int CollectionId, byte[] Data)>();
        foreach (CollectionState collection in collections)
        {
            int id = collection.Definition.Id;
            var definition = new ByteWriter();
            collection.Definition.Write(definition);
            pages.Add((PageCollectionDef, id, definition.ToArray()));

            PersistedNode[] nodes = collection.SnapshotNodes();
            var vectors = new ByteWriter();
            var graph = new ByteWriter();
            vectors.WriteInt32(nodes.Length);
            graph.WriteInt32(nodes.Length);
            foreach (PersistedNode node in nodes)
            {
                // Keep vector bytes and topology in separate pages so mmap reads never pull in
                // adjacency data, and page sizes stay deterministic for identical operation logs.
                vectors.WriteInt32(node.Id);
                vectors.WriteInt32(node.Level);
                vectors.WriteBool(node.Deleted);
                vectors.WriteBlob(node.Vector);

                graph.WriteInt32(node.Id);
                node.WriteLayersOnly(graph);
            }

            pages.Add((PageVector, id, vectors.ToArray()));
            pages.Add((PageGraph, id, graph.ToArray()));

            PersistedRecord[] records = collection.SnapshotRecords();
            var payload = new ByteWriter();
            payload.WriteInt32(records.Length);
            foreach (PersistedRecord record in records)
            {
                record.Write(payload);
            }

            pages.Add((PagePayload, id, payload.ToArray()));

            var text = new ByteWriter();
            Bm25Snapshot snapshot = textSnapshots.TryGetValue(id, out Bm25Snapshot? value)
                ? value
                : Bm25Snapshot.Empty;
            text.WriteInt32(snapshot.Documents.Length);
            foreach (Bm25Snapshot.DocumentSnapshot document in snapshot.Documents)
            {
                text.WriteInt32(document.DocId);
                text.WriteInt32(document.Length);
                text.WriteInt32(document.Terms.Length);
                foreach ((string term, int frequency) in document.Terms)
                {
                    text.WriteString(term);
                    text.WriteInt32(frequency);
                }
            }

            pages.Add((PageText, id, text.ToArray()));
        }

        long pageTableOffset = HeaderSize;
        long dataOffset = Align(HeaderSize + (long)pages.Count * PageRecordSize);
        var table = new byte[(long)pages.Count * PageRecordSize];
        var layout = new List<(byte[] Data, long Offset, int Length)>();
        long cursor = dataOffset;
        for (int i = 0; i < pages.Count; i++)
        {
            (byte kind, int collectionId, byte[] data) = pages[i];
            int tableEntry = i * PageRecordSize;
            table[tableEntry] = kind;
            BinaryPrimitives.WriteInt32LittleEndian(table.AsSpan(tableEntry + 4, 4), collectionId);
            BinaryPrimitives.WriteInt64LittleEndian(table.AsSpan(tableEntry + 8, 8), cursor);
            BinaryPrimitives.WriteInt64LittleEndian(table.AsSpan(tableEntry + 16, 8), data.Length);
            BinaryPrimitives.WriteUInt32LittleEndian(table.AsSpan(tableEntry + 24, 4), Crc32C.Compute(data));
            layout.Add((data, cursor, data.Length));
            cursor = Align(cursor + data.Length);
        }

        string temporary = path + ".tmp";
        using (var stream = new FileStream(
            temporary, FileMode.Create, FileAccess.ReadWrite, FileShare.None,
            bufferSize: 64 * 1024, FileOptions.SequentialScan))
        {
            var header = new byte[HeaderSize];
            Magic.CopyTo(header, 0);
            BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(4, 4), Version);
            BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8, 4), HeaderSize);
            BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(16, 8), databaseId);
            BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(32, 8), (ulong)pageTableOffset);
            BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(40, 4), (uint)pages.Count);
            BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(44, 4), (uint)collections.Length);
            BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(64, 4), Crc32C.Compute(header.AsSpan(0, 64)));

            stream.Write(header);
            stream.Write(table);
            foreach ((byte[] data, long offset, int length) in layout)
            {
                stream.Seek(offset, SeekOrigin.Begin);
                stream.Write(data);
            }

            stream.SetLength(cursor);
            stream.Flush(flushToDisk: true);
        }

        File.Move(temporary, path, overwrite: true);
    }

    /// <summary>Reads and validates a checkpoint file. A missing or invalid file yields empty state.</summary>
    internal static CheckpointData Read(string path)
    {
        if (!File.Exists(path))
        {
            return new CheckpointData(0, [], [], []);
        }

        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length < HeaderSize ||
            !bytes.AsSpan(0, Magic.Length).SequenceEqual(Magic) ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4, 4)) != Version ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(8, 4)) != HeaderSize ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(64, 4)) != Crc32C.Compute(bytes.AsSpan(0, 64)))
        {
            throw new InvalidDataException("not a supported Faro checkpoint (bad header or version)");
        }

        long tableOffset = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(32, 8));
        int pageCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(40, 4)));
        if (tableOffset < HeaderSize || pageCount < 0 || tableOffset + (long)pageCount * PageRecordSize > bytes.Length)
        {
            throw new InvalidDataException("invalid checkpoint page table");
        }

        var definitions = new Dictionary<int, CollectionDefinition>();
        var vectors = new Dictionary<int, List<PersistedNode>>();
        var graph = new Dictionary<int, List<(int Id, int[][] Neighbors, int[][] Backbone)>>();
        var records = new Dictionary<int, PersistedRecord[]>();

        for (int i = 0; i < pageCount; i++)
        {
            int entry = checked((int)tableOffset) + i * PageRecordSize;
            byte kind = bytes[entry];
            int collectionId = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(entry + 4, 4));
            long offset = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(entry + 8, 8));
            long length = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(entry + 16, 8));
            uint checksum = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry + 24, 4));

            if (length < 0 || offset < HeaderSize || offset > bytes.Length - length)
            {
                continue; // a page outside the file is treated as absent
            }

            ReadOnlyMemory<byte> page = new ReadOnlyMemory<byte>(bytes, checked((int)offset), checked((int)length));
            if (Crc32C.Compute(page.Span) != checksum)
            {
                continue; // CRC failure: page is absent (torn or corrupt)
            }

            var reader = new ByteReader(page);
            try
            {
                switch (kind)
                {
                    case PageCollectionDef:
                        CollectionDefinition definition = CollectionDefinition.Read(reader);
                        definitions[definition.Id] = definition;
                        break;

                    case PageVector:
                        ReadVectorPage(reader, collectionId, vectors);
                        break;

                    case PageGraph:
                        ReadGraphPage(reader, collectionId, graph);
                        break;

                    case PagePayload:
                        ReadPayloadPage(reader, collectionId, records);
                        break;

                    case PageText:
                        // Text pages are validated for integrity; BM25 itself is rebuilt from the
                        // typed payloads when a collection is materialised, keeping replay
                        // independent of a user-supplied tokenizer.
                        break;
                }
            }
            catch (InvalidDataException)
            {
                // A malformed but CRC-valid page behaves like an absent page.
            }
        }

        var nodes = new Dictionary<int, PersistedNode[]>();
        foreach ((int id, CollectionDefinition definition) in definitions)
        {
            if (!vectors.TryGetValue(id, out List<PersistedNode>? vectorNodes) ||
                !graph.TryGetValue(id, out List<(int, int[][], int[][])>? graphNodes))
            {
                continue;
            }

            var topology = graphNodes.ToDictionary(
                static x => x.Item1,
                static x => (x.Item2, x.Item3));
            var joined = new List<PersistedNode>(vectorNodes.Count);
            foreach (PersistedNode node in vectorNodes)
            {
                if (!topology.TryGetValue(node.Id, out (int[][] Neighbors, int[][] Backbone) links) ||
                    node.Vector.Length != ExpectedVectorBytes(definition))
                {
                    continue;
                }

                joined.Add(new PersistedNode(
                    node.Id, node.Level, node.Deleted, node.Vector, links.Neighbors, links.Backbone));
            }

            nodes[id] = joined.ToArray();
        }

        return new CheckpointData(
            BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(16, 8)),
            definitions,
            nodes,
            records);
    }

    /// <summary>
    /// Locates every stored vector blob in the checkpoint file. The returned offsets are absolute
    /// file positions and are used to index directly into the memory-mapped vector pages.
    /// </summary>
    internal static Dictionary<int, Dictionary<int, long>> ReadVectorOffsets(string path)
    {
        var result = new Dictionary<int, Dictionary<int, long>>();
        if (!File.Exists(path))
        {
            return result;
        }

        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length < HeaderSize ||
            !bytes.AsSpan(0, Magic.Length).SequenceEqual(Magic) ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4, 4)) != Version ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(8, 4)) != HeaderSize ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(64, 4)) != Crc32C.Compute(bytes.AsSpan(0, 64)))
        {
            return result;
        }

        long tableOffset = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(32, 8));
        int pageCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(40, 4)));
        if (tableOffset < HeaderSize || pageCount < 0 || tableOffset + (long)pageCount * PageRecordSize > bytes.Length)
        {
            return result;
        }

        for (int i = 0; i < pageCount; i++)
        {
            int entry = checked((int)tableOffset) + i * PageRecordSize;
            if (bytes[entry] != PageVector)
            {
                continue;
            }

            int collectionId = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(entry + 4, 4));
            long offset = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(entry + 8, 8));
            long length = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(entry + 16, 8));
            uint checksum = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry + 24, 4));
            if (length < 0 || offset < HeaderSize || offset > bytes.Length - length)
            {
                continue;
            }

            int pageStart = checked((int)offset);
            ReadOnlyMemory<byte> page = new(bytes, pageStart, checked((int)length));
            if (Crc32C.Compute(page.Span) != checksum)
            {
                continue;
            }

            var reader = new ByteReader(page);
            try
            {
                int count = reader.ReadInt32();
                if (count < 0 || count > 50_000_000)
                {
                    continue;
                }

                var offsets = new Dictionary<int, long>();
                for (int node = 0; node < count; node++)
                {
                    int id = reader.ReadInt32();
                    _ = reader.ReadInt32(); // level
                    _ = reader.ReadBool();  // deleted
                    int vectorLength = reader.ReadInt32();
                    if (vectorLength < 0 || id < 0)
                    {
                        throw new InvalidDataException("invalid vector page entry");
                    }

                    offsets[id] = pageStart + reader.Position;
                    _ = reader.ReadRaw(vectorLength);
                }

                result[collectionId] = offsets;
            }
            catch (InvalidDataException)
            {
                // Ignore a malformed vector page; its nodes fall back to in-memory data.
            }
        }

        return result;
    }

    private static int ExpectedVectorBytes(CollectionDefinition definition) =>
        definition.Distance == Distance.Hamming
            ? BitVectors.WordsFor(definition.Dimensions) * sizeof(ulong)
            : definition.Dimensions * sizeof(float);

    private static void ReadVectorPage(ByteReader reader, int collectionId, Dictionary<int, List<PersistedNode>> into)
    {
        int count = reader.ReadInt32();
        if (count < 0 || count > 50_000_000)
        {
            throw new InvalidDataException("invalid node count");
        }

        var nodes = new List<PersistedNode>(count);
        for (int i = 0; i < count; i++)
        {
            int id = reader.ReadInt32();
            int level = reader.ReadInt32();
            bool isDeleted = reader.ReadBool();
            byte[] vector = reader.ReadBlob().ToArray();
            if (id < 0 || id >= count)
            {
                throw new InvalidDataException("node id outside vector page");
            }

            nodes.Add(new PersistedNode(id, level, isDeleted, vector, [], []));
        }

        into[collectionId] = nodes;
    }

    private static void ReadGraphPage(
        ByteReader reader,
        int collectionId,
        Dictionary<int, List<(int Id, int[][] Neighbors, int[][] Backbone)>> into)
    {
        int count = reader.ReadInt32();
        if (count < 0 || count > 50_000_000)
        {
            throw new InvalidDataException("invalid graph node count");
        }

        var result = new List<(int, int[][], int[][])>(count);
        for (int i = 0; i < count; i++)
        {
            int id = reader.ReadInt32();
            int[][] neighbors = ReadLayers(reader);
            int[][] backbone = ReadLayers(reader);
            result.Add((id, neighbors, backbone));
        }

        into[collectionId] = result;
    }

    private static int[][] ReadLayers(ByteReader reader)
    {
        int layers = reader.ReadInt32();
        if (layers < 0 || layers > 25)
        {
            throw new InvalidDataException("invalid layer count");
        }

        var result = new int[layers][];
        for (int i = 0; i < layers; i++)
        {
            int size = reader.ReadInt32();
            if (size < 0 || size > 1_000_000)
            {
                throw new InvalidDataException("invalid adjacency length");
            }

            result[i] = new int[size];
            for (int j = 0; j < size; j++)
            {
                result[i][j] = reader.ReadInt32();
            }
        }

        return result;
    }

    private static void ReadPayloadPage(ByteReader reader, int collectionId, Dictionary<int, PersistedRecord[]> into)
    {
        int count = reader.ReadInt32();
        if (count < 0 || count > 50_000_000)
        {
            throw new InvalidDataException("invalid record count");
        }

        var records = new PersistedRecord[count];
        for (int i = 0; i < count; i++)
        {
            records[i] = PersistedRecord.Read(reader);
        }

        into[collectionId] = records;
    }

    private static long Align(long value) => (value + Alignment - 1) & ~(Alignment - 1L);
}
