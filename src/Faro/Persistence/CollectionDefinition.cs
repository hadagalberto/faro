using Faro.Payload;

namespace Faro.Persistence;

/// <summary>Stored field declaration; the selector is supplied later by the typed schema.</summary>
internal readonly record struct StoredField(string Name, FieldKind Kind);

/// <summary>
/// Complete, serializer-independent description of one collection, persisted in a
/// CollectionDef page and in WAL create-collection records.
/// </summary>
internal sealed class CollectionDefinition
{
    internal CollectionDefinition(
        int id,
        string name,
        KeyKind keyKind,
        int dimensions,
        Distance distance,
        bool normalizeOnAdd,
        int m,
        int efConstruction,
        int efSearch,
        long? seed,
        double? levelMultiplier,
        bool extendCandidates,
        bool keepPrunedConnections,
        double mc,
        QuantizationKind? quantizationKind,
        int rescoreFactor,
        double? quantizationQuantile,
        double bm25K1,
        double bm25B,
        StoredField[] fields,
        string? fullTextField)
    {
        Id = id;
        Name = name;
        KeyKind = keyKind;
        Dimensions = dimensions;
        Distance = distance;
        NormalizeOnAdd = normalizeOnAdd;
        M = m;
        EfConstruction = efConstruction;
        EfSearch = efSearch;
        Seed = seed;
        LevelMultiplier = levelMultiplier;
        ExtendCandidates = extendCandidates;
        KeepPrunedConnections = keepPrunedConnections;
        Mc = mc;
        QuantizationKind = quantizationKind;
        RescoreFactor = rescoreFactor;
        QuantizationQuantile = quantizationQuantile;
        Bm25K1 = bm25K1;
        Bm25B = bm25B;
        Fields = fields;
        FullTextField = fullTextField;
    }

    internal int Id { get; }
    internal string Name { get; }
    internal KeyKind KeyKind { get; }
    internal int Dimensions { get; }
    internal Distance Distance { get; }
    internal bool NormalizeOnAdd { get; }
    internal int M { get; }
    internal int EfConstruction { get; }
    internal int EfSearch { get; }
    internal long? Seed { get; }
    internal double? LevelMultiplier { get; }
    internal bool ExtendCandidates { get; }
    internal bool KeepPrunedConnections { get; }
    internal double Mc { get; }
    internal QuantizationKind? QuantizationKind { get; }
    internal int RescoreFactor { get; }
    internal double? QuantizationQuantile { get; }
    internal double Bm25K1 { get; }
    internal double Bm25B { get; }
    internal StoredField[] Fields { get; }
    internal string? FullTextField { get; }

    internal HnswOptions ToHnswOptions() => new()
    {
        Dimensions = Dimensions,
        Distance = Distance,
        NormalizeOnAdd = NormalizeOnAdd,
        M = M,
        EfConstruction = EfConstruction,
        EfSearch = EfSearch,
        Seed = Seed,
        LevelMultiplier = LevelMultiplier,
        ExtendCandidates = ExtendCandidates,
        KeepPrunedConnections = KeepPrunedConnections,
        Mc = Mc,
        Quantization = MakeQuantization(),
    };

    internal QuantizationOptions? MakeQuantization() => QuantizationKind switch
    {
        null => null,
        Faro.QuantizationKind.Int8 => Quantization.Int8(RescoreFactor, QuantizationQuantile),
        Faro.QuantizationKind.Binary => Quantization.Binary(RescoreFactor),
        _ => throw new InvalidDataException($"unknown quantization kind {(int)QuantizationKind!.Value}"),
    };

    internal void Write(ByteWriter writer)
    {
        writer.WriteInt32(Id);
        writer.WriteString(Name);
        writer.WriteByte((byte)KeyKind);
        writer.WriteInt32(Dimensions);
        writer.WriteByte((byte)Distance);
        writer.WriteBool(NormalizeOnAdd);
        writer.WriteInt32(M);
        writer.WriteInt32(EfConstruction);
        writer.WriteInt32(EfSearch);
        writer.WriteNullableLong(Seed);
        writer.WriteNullableDouble(LevelMultiplier);
        writer.WriteBool(ExtendCandidates);
        writer.WriteBool(KeepPrunedConnections);
        writer.WriteDouble(Mc);
        writer.WriteByte(QuantizationKind is null ? (byte)0 : (byte)(QuantizationKind.Value + 1));
        writer.WriteInt32(RescoreFactor);
        writer.WriteNullableDouble(QuantizationQuantile);
        writer.WriteDouble(Bm25K1);
        writer.WriteDouble(Bm25B);
        writer.WriteInt32(Fields.Length);
        foreach (StoredField field in Fields)
        {
            writer.WriteString(field.Name);
            writer.WriteByte((byte)field.Kind);
        }

        writer.WriteNullableString(FullTextField);
    }

    internal static CollectionDefinition Read(ByteReader reader)
    {
        int id = reader.ReadInt32();
        string name = reader.ReadString();
        var keyKind = (KeyKind)reader.ReadByte();
        int dimensions = reader.ReadInt32();
        var distance = (Distance)reader.ReadByte();
        bool normalize = reader.ReadBool();
        int m = reader.ReadInt32();
        int efConstruction = reader.ReadInt32();
        int efSearch = reader.ReadInt32();
        long? seed = reader.ReadNullableLong();
        double? levelMultiplier = reader.ReadNullableDouble();
        bool extend = reader.ReadBool();
        bool keepPruned = reader.ReadBool();
        double mc = reader.ReadDouble();
        byte kindByte = reader.ReadByte();
        QuantizationKind? quantization = kindByte == 0 ? null : (QuantizationKind)(kindByte - 1);
        int rescore = reader.ReadInt32();
        double? quantile = reader.ReadNullableDouble();
        double k1 = reader.ReadDouble();
        double b = reader.ReadDouble();
        int fieldCount = reader.ReadInt32();
        if (fieldCount < 0 || fieldCount > 1024)
        {
            throw new InvalidDataException("invalid payload field count");
        }

        var fields = new StoredField[fieldCount];
        for (int i = 0; i < fieldCount; i++)
        {
            fields[i] = new StoredField(reader.ReadString(), (FieldKind)reader.ReadByte());
        }

        string? fullText = reader.ReadNullableString();

        if (dimensions <= 0 || m < 2 || efConstruction < 1 || efSearch < 1)
        {
            throw new InvalidDataException("invalid collection definition");
        }

        if (!Enum.IsDefined(keyKind) || !Enum.IsDefined(distance) ||
            (quantization.HasValue && !Enum.IsDefined(quantization.Value)))
        {
            throw new InvalidDataException("unknown value in collection definition");
        }

        foreach (StoredField field in fields)
        {
            if (!Enum.IsDefined(field.Kind))
            {
                throw new InvalidDataException("unknown payload field kind");
            }
        }

        return new CollectionDefinition(
            id, name, keyKind, dimensions, distance, normalize, m, efConstruction, efSearch,
            seed, levelMultiplier, extend, keepPruned, mc, quantization, rescore, quantile,
            k1, b, fields, fullText);
    }
}
