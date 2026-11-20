namespace Faro.Persistence;

/// <summary>How <see cref="FaroDb"/> reacts to damaged data in a checkpoint or WAL.</summary>
public enum CheckpointRecovery
{
    /// <summary>Default. Raise <see cref="FaroCorruptionException"/> instead of silently dropping data.</summary>
    Fail = 0,

    /// <summary>Opt in to salvaging undamaged data. Everything skipped is reported on <see cref="FaroDb.RecoveryReport"/>.</summary>
    Salvage = 1,
}

/// <summary>One damaged region that recovery skipped (or refused to open).</summary>
/// <param name="PageKind">Checkpoint page kind byte, or <c>0</c> for a WAL/header problem.</param>
/// <param name="CollectionId">Owning collection id, or <c>-1</c> when unknown.</param>
/// <param name="Offset">Absolute file offset of the damaged region.</param>
/// <param name="Length">Declared length of the damaged region.</param>
/// <param name="Reason">Human-readable cause (bad checksum, out of bounds, malformed page, ...).</param>
public sealed record CorruptionIssue(byte PageKind, int CollectionId, long Offset, long Length, string Reason);

/// <summary>What recovery skipped when a database was opened with <see cref="CheckpointRecovery.Salvage"/>.</summary>
public sealed class CorruptionReport
{
    private readonly List<CorruptionIssue> _issues = [];

    internal void Add(byte pageKind, int collectionId, long offset, long length, string reason) =>
        _issues.Add(new CorruptionIssue(pageKind, collectionId, offset, length, reason));

    internal void Add(CorruptionIssue issue) => _issues.Add(issue);

    /// <summary>Damaged regions, in the order they were encountered.</summary>
    public IReadOnlyList<CorruptionIssue> Issues => _issues;

    /// <summary>Whether any damage was found.</summary>
    public bool HasIssues => _issues.Count > 0;
}

/// <summary>
/// Raised when a checkpoint page or WAL structure fails validation. Derives from
/// <see cref="IOException"/> (the file is unreadable as a database), while callers that care can
/// distinguish corruption explicitly.
/// </summary>
public sealed class FaroCorruptionException : IOException
{
    /// <summary>Creates a corruption error with a diagnostic message.</summary>
    public FaroCorruptionException(string message) : base(message) { }

    /// <summary>Creates a corruption error with a diagnostic message and underlying cause.</summary>
    public FaroCorruptionException(string message, Exception innerException) : base(message, innerException) { }
}
