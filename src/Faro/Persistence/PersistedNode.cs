namespace Faro.Persistence;

/// <summary>A checkpoint/WAL snapshot of one HNSW node: stored vector, level and topology.</summary>
internal sealed class PersistedNode
{
    internal PersistedNode(int id, int level, bool deleted, byte[] vector, int[][] neighbors, int[][] backbone)
    {
        Id = id;
        Level = level;
        Deleted = deleted;
        Vector = vector;
        Neighbors = neighbors;
        Backbone = backbone;
    }

    internal int Id { get; }
    internal int Level { get; }
    internal bool Deleted { get; }
    internal byte[] Vector { get; }
    internal int[][] Neighbors { get; }
    internal int[][] Backbone { get; }

    internal void Write(ByteWriter writer)
    {
        writer.WriteInt32(Id);
        writer.WriteInt32(Level);
        writer.WriteBool(Deleted);
        writer.WriteBlob(Vector);
        WriteLayers(writer, Neighbors);
        WriteLayers(writer, Backbone);
    }

    internal static PersistedNode Read(ByteReader reader)
    {
        int id = reader.ReadInt32();
        int level = reader.ReadInt32();
        bool deleted = reader.ReadBool();
        byte[] vector = reader.ReadBlob().ToArray();
        int[][] neighbors = ReadLayers(reader);
        int[][] backbone = ReadLayers(reader);
        if (id < 0 || level < 0 || level > 24 || vector.Length == 0)
        {
            throw new InvalidDataException("invalid persisted node");
        }

        return new PersistedNode(id, level, deleted, vector, neighbors, backbone);
    }

    internal PersistedNode WithDeleted(bool deleted) =>
        new(Id, Level, deleted, Vector, Neighbors, Backbone);

    /// <summary>Writes one node's topology arrays without the vector blob (graph page layout).</summary>
    internal void WriteLayersOnly(ByteWriter writer)
    {
        WriteLayers(writer, Neighbors);
        WriteLayers(writer, Backbone);
    }

    private static void WriteLayers(ByteWriter writer, int[][] layers)
    {
        writer.WriteInt32(layers.Length);
        foreach (int[] layer in layers)
        {
            writer.WriteInt32(layer.Length);
            foreach (int id in layer)
            {
                writer.WriteInt32(id);
            }
        }
    }

    private static int[][] ReadLayers(ByteReader reader)
    {
        int count = reader.ReadInt32();
        if (count < 0 || count > 25)
        {
            throw new InvalidDataException("invalid layer count");
        }

        var layers = new int[count][];
        for (int i = 0; i < count; i++)
        {
            int size = reader.ReadInt32();
            if (size < 0 || size > 1_000_000)
            {
                throw new InvalidDataException("invalid adjacency length");
            }

            layers[i] = new int[size];
            for (int j = 0; j < size; j++)
            {
                layers[i][j] = reader.ReadInt32();
            }
        }

        return layers;
    }
}
