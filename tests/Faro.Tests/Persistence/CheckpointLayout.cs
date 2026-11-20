using System.Buffers.Binary;

namespace Faro.Tests.Persistence;

/// <summary>
/// Independent reader for the documented checkpoint page table (docs/file-format.md). Used by
/// corruption tests so expected page locations never come from the library under test.
/// </summary>
internal readonly record struct CheckpointPage(byte Kind, int CollectionId, long Offset, long Length, uint Crc);

internal static class CheckpointLayout
{
    internal const int HeaderSize = 128;
    internal const int PageRecordSize = 32;

    internal static CheckpointPage[] Pages(byte[] file)
    {
        if (file.Length < HeaderSize)
        {
            throw new InvalidDataException("file shorter than the checkpoint header");
        }

        long tableOffset = BinaryPrimitives.ReadInt64LittleEndian(file.AsSpan(32, 8));
        int count = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(file.AsSpan(40, 4)));
        var result = new CheckpointPage[count];
        for (int i = 0; i < count; i++)
        {
            int entry = checked((int)tableOffset) + i * PageRecordSize;
            result[i] = new CheckpointPage(
                file[entry],
                BinaryPrimitives.ReadInt32LittleEndian(file.AsSpan(entry + 4, 4)),
                BinaryPrimitives.ReadInt64LittleEndian(file.AsSpan(entry + 8, 8)),
                BinaryPrimitives.ReadInt64LittleEndian(file.AsSpan(entry + 16, 8)),
                BinaryPrimitives.ReadUInt32LittleEndian(file.AsSpan(entry + 24, 4)));
        }

        return result;
    }
}
