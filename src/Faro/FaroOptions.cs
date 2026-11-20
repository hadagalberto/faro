namespace Faro;

/// <summary>Options for opening or creating a <see cref="FaroDb"/>.</summary>
public sealed class FaroOptions
{
    /// <summary>WAL flush policy. Defaults to <see cref="Durability.Batched"/>.</summary>
    public Durability Durability { get; set; } = Durability.Batched;

    /// <summary>
    /// When <see langword="true"/>, a torn WAL tail is truncated so new records can be appended.
    /// Defaults to <see langword="true"/>.
    /// </summary>
    public bool TruncateTornTail { get; set; } = true;
}
