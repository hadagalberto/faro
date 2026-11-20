using Faro.Filtering;
using Microsoft.Extensions.VectorData;

namespace Faro.VectorData;

/// <summary>
/// Adapter for <c>GetDynamicCollection</c>: the abstractions type dynamic keys as <see cref="object"/>,
/// while Faro persists keys as <see cref="string"/>. Keys are validated and converted at the boundary.
/// </summary>
internal sealed class FaroDynamicKeyedCollection<TRecord> : IFaroKeyedCollection<object, TRecord>
    where TRecord : class
{
    private readonly FaroDbCollection<string, TRecord> _inner;

    private FaroDynamicKeyedCollection(FaroDbCollection<string, TRecord> inner) => _inner = inner;

    public static FaroDynamicKeyedCollection<TRecord> Create(
        FaroDb db, string name, CollectionOptions<TRecord> options) =>
        new(db.GetOrCreateCollection<string, TRecord>(name, options));

    public string Name => _inner.Name;

    public bool TryGet(object key, out TRecord? record) => _inner.TryGet(AsString(key), out record);

    public IReadOnlyList<(object Key, TRecord Record)> GetByFilter(Filter filter, int top) =>
        _inner.GetByFilter(filter, top).Select(row => ((object)row.Key, row.Payload)).ToArray();

    public bool Delete(object key) => _inner.Delete(AsString(key));

    public void Upsert(object key, float[] vector, TRecord record) =>
        _inner.Upsert(AsString(key), vector, record);

    public void UpsertBatch(IEnumerable<(object Key, float[] Vector, TRecord Record)> items) =>
        _inner.UpsertBatch(items.Select(item => (AsString(item.Key), item.Vector, item.Record)));

    public SearchHit<object>[] Search(float[] query, int top, Filter? filter) =>
        _inner.Search(query, top, filter)
            .Select(hit => new SearchHit<object> { Key = hit.Key, Score = hit.Score })
            .ToArray();

    public HybridHit<object>[] HybridSearch(float[] query, string text, int top, Filter? filter) =>
        _inner.HybridSearch(query, text, top, filter: filter)
            .Select(hit => new HybridHit<object>
            {
                Key = hit.Key,
                Score = hit.Score,
                VectorScore = hit.VectorScore,
                TextScore = hit.TextScore,
                VectorRank = hit.VectorRank,
                TextRank = hit.TextRank,
            })
            .ToArray();

    private static string AsString(object key) => key as string
        ?? throw new VectorStoreException("dynamic collections use string keys")
        {
            OperationName = "KeyConversion",
        };
}
