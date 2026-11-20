namespace Faro;

/// <summary>Controls when committed WAL records are flushed to stable storage.</summary>
public enum Durability
{
    /// <summary>Fsync after every WAL record. Slowest, strongest guarantee.</summary>
    Always = 0,

    /// <summary>Fsync once per commit group (batch, explicit flush, checkpoint or dispose). Default.</summary>
    Batched = 1,

    /// <summary>
    /// No fsync at all. Records remain in the FileStream buffer and the OS page cache, so a
    /// process crash (<c>Environment.FailFast</c>, <c>kill</c>) or power failure can lose recent
    /// commits — including records from a completed batch that sat below the internal flush
    /// threshold. Use only for data that can be rebuilt from another source.
    /// </summary>
    None = 2,
}
