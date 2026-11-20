namespace Faro.Hybrid;

/// <summary>Serializable snapshot of a BM25 inverted index for the checkpoint text page.</summary>
internal sealed class Bm25Snapshot
{
    internal Bm25Snapshot(DocumentSnapshot[] documents)
    {
        Documents = documents;
    }

    internal DocumentSnapshot[] Documents { get; }

    internal readonly record struct DocumentSnapshot(int DocId, int Length, (string Term, int Frequency)[] Terms);

    internal static Bm25Snapshot Empty => new([]);
}
