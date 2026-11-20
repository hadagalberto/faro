using Faro.Filtering;

namespace Faro.VectorData;

/// <summary>Key-shape-independent view over a Faro collection used by the connector.</summary>
internal interface IFaroKeyedCollection<TKey, TRecord>
    where TKey : notnull
    where TRecord : class
{
    string Name { get; }

    bool TryGet(TKey key, out TRecord? record);

    IReadOnlyList<(TKey Key, TRecord Record)> GetByFilter(Filter filter, int top);

    bool Delete(TKey key);

    void Upsert(TKey key, float[] vector, TRecord record);

    void UpsertBatch(IEnumerable<(TKey Key, float[] Vector, TRecord Record)> items);

    SearchHit<TKey>[] Search(float[] query, int top, Filter? filter);

    HybridHit<TKey>[] HybridSearch(float[] query, string text, int top, Filter? filter);
}
