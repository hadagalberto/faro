using Faro.Persistence;

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

    /// <summary>
    /// How damaged checkpoint pages and WAL headers are handled. Defaults to <see cref="CheckpointRecovery.Fail"/>:
    /// corruption raises <see cref="FaroCorruptionException"/> instead of being treated as absent data.
    /// Set to <see cref="CheckpointRecovery.Salvage"/> (or use <see cref="FaroDb.OpenWithRecovery"/>) to
    /// load everything that is still valid and inspect <see cref="FaroDb.RecoveryReport"/>.
    /// </summary>
    public CheckpointRecovery Recovery { get; set; } = CheckpointRecovery.Fail;
}
