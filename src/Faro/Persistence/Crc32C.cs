namespace Faro.Persistence;

/// <summary>
/// CRC32C (Castagnoli) checksum, reflected form, polynomial 0x82F63B78.
/// </summary>
/// <remarks>
/// Castagnoli et al., "Optimization of Cyclic Redundancy-Check Codes with 24 and 32 Parity Bits"
/// (1993). The reflected table is generated at startup; no hardware CRC instruction is required,
/// so the implementation is identical on every target and AOT-safe.
/// </remarks>
internal static class Crc32C
{
    private static readonly uint[] Table = BuildTable();

    /// <summary>Computes the CRC32C of a byte span.</summary>
    internal static uint Compute(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFFu;
        for (int i = 0; i < data.Length; i++)
        {
            crc = Table[(byte)(crc ^ data[i])] ^ (crc >> 8);
        }

        return crc ^ 0xFFFFFFFFu;
    }

    /// <summary>Computes the CRC32C of the concatenation of two spans.</summary>
    internal static uint Compute(ReadOnlySpan<byte> first, ReadOnlySpan<byte> second)
    {
        uint crc = 0xFFFFFFFFu;
        for (int i = 0; i < first.Length; i++)
        {
            crc = Table[(byte)(crc ^ first[i])] ^ (crc >> 8);
        }

        for (int i = 0; i < second.Length; i++)
        {
            crc = Table[(byte)(crc ^ second[i])] ^ (crc >> 8);
        }

        return crc ^ 0xFFFFFFFFu;
    }

    private static uint[] BuildTable()
    {
        var table = new uint[256];
        for (uint i = 0; i < table.Length; i++)
        {
            uint value = i;
            for (int bit = 0; bit < 8; bit++)
            {
                value = (value & 1) != 0 ? (value >> 1) ^ 0x82F63B78u : value >> 1;
            }

            table[i] = value;
        }

        return table;
    }
}
