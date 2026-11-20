namespace Faro;

/// <summary>Controls when committed WAL records are flushed to stable storage.</summary>
public enum Durability
{
    /// <summary>Fsync after every WAL record. Slowest, strongest guarantee.</summary>
    Always = 0,

    /// <summary>Fsync once per commit group (batch, explicit flush, checkpoint or dispose). Default.</summary>
    Batched = 1,

    /// <summary>Rely on the OS page cache; recent commits may be lost on power failure.</summary>
    None = 2,
}
