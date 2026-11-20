using Faro.Filtering;

namespace Faro.VectorData;

/// <summary>Direct adapter for Faro's supported key types.</summary>
internal sealed class FaroKeyedCollection<TKey, TRecord> : IFaroKeyedCollection<TKey, TRecord>
    where TKey : notnull
    where TRecord : class
{
    private readonly FaroDbCollection<TKey, TRecord> _inner;

    public FaroKeyedCollection(FaroDbCollection<TKey, TRecord> inner) => _inner = inner;

    public string Name => _inner.Name;

    public bool TryGet(TKey key, out TRecord? record) => _inner.TryGet(key, out record);

    public IReadOnlyList<(TKey Key, TRecord Record)> GetByFilter(Filter filter, int top) =>
        _inner.GetByFilter(filter, top);

    public bool Delete(TKey key) => _inner.Delete(key);

    public void Upsert(TKey key, float[] vector, TRecord record) => _inner.Upsert(key, vector, record);

    public void UpsertBatch(IEnumerable<(TKey Key, float[] Vector, TRecord Record)> items) =>
        _inner.UpsertBatch(items.Select(item => (item.Key, item.Vector, item.Record)));

    public SearchHit<TKey>[] Search(float[] query, int top, Filter? filter) =>
        _inner.Search(query, top, filter);

    public HybridHit<TKey>[] HybridSearch(float[] query, string text, int top, Filter? filter) =>
        _inner.HybridSearch(query, text, top, filter: filter);
}
