using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using Faro.Filtering;
using Faro.Hybrid;
using Faro.Payload;
using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;
using FaroFilter = Faro.Filtering.Filter;

namespace Faro.VectorData;

/// <summary>
/// A <see cref="VectorStoreCollection{TKey,TRecord}"/> backed by a Faro collection, including
/// <see cref="IKeywordHybridSearchable{TRecord}"/> for vector + BM25 retrieval.
/// </summary>
public sealed class FaroVectorStoreCollection<TKey, TRecord> :
    VectorStoreCollection<TKey, TRecord>, IKeywordHybridSearchable<TRecord>
    where TKey : notnull
    where TRecord : class
{
    private readonly FaroVectorStore _store;
    private readonly CollectionModel _model;
    private readonly KeyPropertyModel _keyProperty;
    private readonly VectorPropertyModel _vectorProperty;
    private readonly DataPropertyModel? _textProperty;
    private readonly IReadOnlyDictionary<string, FieldKind> _fields;
    private readonly Distance _distance;
    private readonly string _distanceFunction;
    private readonly CollectionOptions<TRecord> _faroOptions;

    private IFaroKeyedCollection<TKey, TRecord> _collection;
    private bool _deleted;

    internal FaroVectorStoreCollection(
        FaroVectorStore store,
        FaroDb db,
        string name,
        CollectionModel model,
        IPayloadSerializer<TRecord> serializer,
        FaroVectorStoreOptions options)
    {
        _store = store;
        _model = model;
        _keyProperty = model.KeyProperty;
        _vectorProperty = model.VectorProperty;
        _textProperty = model.DataProperties.FirstOrDefault(property => property.IsFullTextIndexed);

        _distanceFunction = string.IsNullOrWhiteSpace(_vectorProperty.DistanceFunction)
            ? FaroVectorDataMapping.DefaultDistanceFunction(options.Index?.Distance ?? Distance.Cosine)
            : _vectorProperty.DistanceFunction!;
        FaroVectorDataMapping.ValidateIndexKind(_vectorProperty.IndexKind, name);
        _distance = FaroVectorDataMapping.ToDistance(_distanceFunction, name);

        PayloadSchema<TRecord> schema = FaroSchemaFactory.Create<TRecord>(model, name);
        _fields = schema.FieldNames.ToDictionary(
            field => field,
            schema.KindOf,
            StringComparer.Ordinal);

        _faroOptions = new CollectionOptions<TRecord>
        {
            Dimensions = _vectorProperty.Dimensions,
            Distance = _distance,
            Index = options.Index,
            Quantization = options.Quantization,
            Schema = schema,
            Serializer = serializer,
            DefaultEfSearch = options.DefaultEfSearch,
        };

        _collection = OpenCollection(db, name);
    }

    /// <inheritdoc />
    public override string Name => _collection.Name;

    /// <inheritdoc />
    public override Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(!_deleted && _store.Database.CollectionNames.Contains(Name));
    }

    /// <inheritdoc />
    public override Task EnsureCollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_deleted)
        {
            _collection = OpenCollection(_store.Database, Name);
            _deleted = false;
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task EnsureCollectionDeletedAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _store.Database.DeleteCollection(Name);
        _deleted = true;
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task<TRecord?> GetAsync(
        TKey key, RecordRetrievalOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ThrowIfDeleted();
        if (!_collection.TryGet(key, out TRecord? record) || record is null)
        {
            return Task.FromResult<TRecord?>(null);
        }

        StripVectors(record, options?.IncludeVectors ?? true);
        return Task.FromResult<TRecord?>(record);
    }

    /// <inheritdoc />
    public override IAsyncEnumerable<TRecord> GetAsync(
        Expression<Func<TRecord, bool>> filter,
        int top,
        FilteredRecordRetrievalOptions<TRecord>? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(top);
        if (options?.Skip is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Skip must not be negative.");
        }

        ThrowIfDeleted();
        FaroFilter faroFilter = Translate(filter);
        int skip = options?.Skip ?? 0;
        var rows = new List<(TKey Key, TRecord Record)>(_collection.GetByFilter(faroFilter, top + skip));

        if (options?.OrderBy is { } orderBy)
        {
            SortRows(rows, orderBy(new FilteredRecordRetrievalOptions<TRecord>.OrderByDefinition()).Values);
        }

        return ToAsyncEnumerable(
            rows.Skip(skip).Take(top).Select(row =>
            {
                StripVectors(row.Record, options?.IncludeVectors ?? true);
                return row.Record;
            }),
            cancellationToken);
    }

    /// <inheritdoc />
    public override Task DeleteAsync(TKey key, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ThrowIfDeleted();
        _collection.Delete(key);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override async Task UpsertAsync(TRecord record, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(record);
        cancellationToken.ThrowIfCancellationRequested();
        ThrowIfDeleted();

        TKey key = ReadKey(record);
        float[] vector = await ReadVectorAsync(record, cancellationToken).ConfigureAwait(false);
        _collection.Upsert(key, vector, record);
    }

    /// <inheritdoc />
    public override async Task UpsertAsync(
        IEnumerable<TRecord> records, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);
        cancellationToken.ThrowIfCancellationRequested();
        ThrowIfDeleted();

        var prepared = new List<(TKey Key, float[] Vector, TRecord Record)>();
        foreach (TRecord record in records)
        {
            if (record is null)
            {
                throw new VectorStoreException("a batch cannot contain null records")
                {
                    CollectionName = Name,
                };
            }

            prepared.Add((ReadKey(record), await ReadVectorAsync(record, cancellationToken).ConfigureAwait(false), record));
        }

        _collection.UpsertBatch(prepared);
    }

    /// <inheritdoc />
    public override IAsyncEnumerable<VectorSearchResult<TRecord>> SearchAsync<TInput>(
        TInput searchValue,
        int top,
        VectorSearchOptions<TRecord>? options = null,
        CancellationToken cancellationToken = default) =>
        SearchCoreAsync(searchValue, top, options, cancellationToken);

    IAsyncEnumerable<VectorSearchResult<TRecord>> IKeywordHybridSearchable<TRecord>.HybridSearchAsync<TInput>(
        TInput searchValue,
        ICollection<string> keywords,
        int top,
        HybridSearchOptions<TRecord>? options,
        CancellationToken cancellationToken) =>
        HybridSearchCoreAsync(searchValue, keywords, top, options, cancellationToken);

    /// <inheritdoc />
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceType == typeof(VectorStoreCollectionMetadata))
        {
            return new VectorStoreCollectionMetadata
            {
                VectorStoreSystemName = "faro",
                VectorStoreName = _store.Database.Path ?? "in-memory",
                CollectionName = Name,
            };
        }

        if (serviceType == typeof(VectorStoreMetadata))
        {
            return _store.GetService(serviceType, serviceKey);
        }

        return null;
    }

    private async IAsyncEnumerable<VectorSearchResult<TRecord>> SearchCoreAsync<TInput>(
        TInput searchValue,
        int top,
        VectorSearchOptions<TRecord>? options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(top);
        if (options?.Skip is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Skip must not be negative.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        ThrowIfDeleted();

        VectorPropertyModel vectorProperty = _model.GetVectorPropertyOrSingle(
            options ?? new VectorSearchOptions<TRecord>());
        EnsureKnownVectorProperty(vectorProperty);

        float[] vector = await FaroVectorConverter.ToVectorAsync(
            searchValue, vectorProperty, Name, cancellationToken).ConfigureAwait(false);
        FaroFilter filter = options?.Filter is null ? FaroFilter.All : Translate(options.Filter);
        SearchHit<TKey>[] hits = _collection.Search(vector, AddSkip(top, options?.Skip ?? 0), filter);

        foreach (VectorSearchResult<TRecord> result in Materialize(
                     hits, options?.ScoreThreshold, options?.Skip ?? 0, top, options?.IncludeVectors ?? true, higherIsBetter: true))
        {
            yield return result;
        }
    }

    private async IAsyncEnumerable<VectorSearchResult<TRecord>> HybridSearchCoreAsync<TInput>(
        TInput searchValue,
        ICollection<string> keywords,
        int top,
        HybridSearchOptions<TRecord>? options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(keywords);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(top);
        if (keywords.Count == 0)
        {
            throw new ArgumentException("hybrid search needs at least one keyword", nameof(keywords));
        }

        if (options?.Skip is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Skip must not be negative.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        ThrowIfDeleted();
        if (_textProperty is null)
        {
            throw FaroVectorDataMapping.Unsupported(Name, "hybrid search without a full-text indexed field");
        }

        if (options?.AdditionalProperty is { } textSelector)
        {
            DataPropertyModel selected = _model.GetFullTextDataPropertyOrSingle(textSelector);
            if (!selected.IsFullTextIndexed || FaroVectorDataMapping.FieldName(selected) != FaroVectorDataMapping.FieldName(_textProperty))
            {
                throw FaroVectorDataMapping.Unsupported(Name, "a full-text property other than the schema's full-text field");
            }
        }

        VectorPropertyModel vectorProperty = _model.GetVectorPropertyOrSingle(
            new VectorSearchOptions<TRecord> { VectorProperty = options?.VectorProperty });
        EnsureKnownVectorProperty(vectorProperty);

        float[] vector = await FaroVectorConverter.ToVectorAsync(
            searchValue, vectorProperty, Name, cancellationToken).ConfigureAwait(false);
        FaroFilter filter = options?.Filter is null ? FaroFilter.All : Translate(options.Filter);
        string text = string.Join(' ', keywords);
        HybridHit<TKey>[] hits = _collection.HybridSearch(vector, text, AddSkip(top, options?.Skip ?? 0), filter: filter);

        foreach (VectorSearchResult<TRecord> result in MaterializeHybrid(
                     hits, options?.ScoreThreshold, options?.Skip ?? 0, top, options?.IncludeVectors ?? true))
        {
            yield return result;
        }
    }

    private IEnumerable<VectorSearchResult<TRecord>> Materialize(
        SearchHit<TKey>[] hits,
        double? threshold,
        int skip,
        int top,
        bool includeVectors,
        bool higherIsBetter)
    {
        int emitted = 0;
        foreach (SearchHit<TKey> hit in hits)
        {
            double score = FaroVectorDataMapping.ToScore(hit.Score, _distanceFunction, _distance);
            if (threshold is { } limit && (higherIsBetter ? score < limit : score > limit))
            {
                continue;
            }

            if (emitted++ < skip)
            {
                continue;
            }

            if (!TryMaterialize(hit.Key, includeVectors, out TRecord? record))
            {
                continue;
            }

            yield return new VectorSearchResult<TRecord>(record!, score);
            if (emitted - skip >= top)
            {
                yield break;
            }
        }
    }

    private IEnumerable<VectorSearchResult<TRecord>> MaterializeHybrid(
        HybridHit<TKey>[] hits,
        double? threshold,
        int skip,
        int top,
        bool includeVectors)
    {
        int emitted = 0;
        foreach (HybridHit<TKey> hit in hits)
        {
            if (threshold is { } limit && hit.Score < limit)
            {
                continue;
            }

            if (emitted++ < skip)
            {
                continue;
            }

            if (!TryMaterialize(hit.Key, includeVectors, out TRecord? record))
            {
                continue;
            }

            yield return new VectorSearchResult<TRecord>(record!, hit.Score);
            if (emitted - skip >= top)
            {
                yield break;
            }
        }
    }

    private bool TryMaterialize(TKey key, bool includeVectors, out TRecord? record)
    {
        if (!_collection.TryGet(key, out record) || record is null)
        {
            record = null;
            return false;
        }

        StripVectors(record, includeVectors);
        return true;
    }

    private TKey ReadKey(TRecord record)
    {
        object? value = _keyProperty.GetValueAsObject(record);
        if (value is not TKey key || key is null)
        {
            throw new VectorStoreException($"record key property '{_keyProperty.ModelName}' is missing or has the wrong type")
            {
                CollectionName = Name,
                OperationName = "Upsert",
            };
        }

        return key;
    }

    private ValueTask<float[]> ReadVectorAsync(TRecord record, CancellationToken cancellationToken) =>
        FaroVectorConverter.ToVectorAsync(
            _vectorProperty.GetValueAsObject(record), _vectorProperty, Name, cancellationToken);

    private FaroFilter Translate(Expression<Func<TRecord, bool>> filter) =>
        new FaroFilterTranslator<TRecord>(_fields, Name).Translate(filter, _model);

    private void EnsureKnownVectorProperty(VectorPropertyModel property)
    {
        if (!ReferenceEquals(property, _vectorProperty) && property.ModelName != _vectorProperty.ModelName)
        {
            throw FaroVectorDataMapping.Unsupported(Name, $"vector property '{property.ModelName}'");
        }
    }

    private void StripVectors(TRecord record, bool includeVectors)
    {
        if (includeVectors)
        {
            return;
        }

        // ProviderServices normalizes the vector model Type to the embedding shape
        // (ReadOnlyMemory<float>) while the accessor targets the CLR property type (for example
        // float[]). Try the supported empty shapes in order so POCOs and dynamic records both work.
        object?[] candidates =
        [
            Array.Empty<float>(),
            default(ReadOnlyMemory<float>),
            default(Memory<float>),
            null,
        ];
        foreach (object? candidate in candidates)
        {
            try
            {
                _vectorProperty.SetValueAsObject(record, candidate);
                return;
            }
            catch (ArgumentException)
            {
                // Try the next supported vector shape.
            }
        }

        throw FaroVectorDataMapping.Unsupported(Name, $"clearing vector property '{_vectorProperty.ModelName}'");
    }

    private void SortRows(List<(TKey Key, TRecord Record)> rows, IReadOnlyList<FilteredRecordRetrievalOptions<TRecord>.OrderByDefinition.SortInfo> sortings)
    {
        if (sortings.Count == 0)
        {
            return;
        }

        var keys = sortings
            .Select(sort => (Property: _model.GetDataOrKeyProperty(sort.PropertySelector), sort.Ascending))
            .ToArray();

        rows.Sort((left, right) =>
        {
            foreach ((PropertyModel property, bool ascending) in keys)
            {
                object? a = property.GetValueAsObject(left.Record);
                object? b = property.GetValueAsObject(right.Record);
                int comparison = a is null ? (b is null ? 0 : -1) : b is null ? 1 : Comparer<object>.Default.Compare(a, b);
                if (comparison != 0)
                {
                    return ascending ? comparison : -comparison;
                }
            }

            return 0;
        });
    }

    private IFaroKeyedCollection<TKey, TRecord> OpenCollection(FaroDb db, string name)
    {
        try
        {
            if (typeof(TKey) == typeof(object))
            {
                return (IFaroKeyedCollection<TKey, TRecord>)(object)FaroDynamicKeyedCollection<TRecord>.Create(
                    db, name, _faroOptions);
            }

            return new FaroKeyedCollection<TKey, TRecord>(db.GetOrCreateCollection<TKey, TRecord>(name, _faroOptions));
        }
        catch (NotSupportedException exception)
        {
            throw new VectorStoreException($"key type '{typeof(TKey)}' is not supported by Faro", exception)
            {
                CollectionName = name,
            };
        }
        catch (InvalidOperationException exception)
        {
            throw new VectorStoreException(exception.Message, exception)
            {
                CollectionName = name,
            };
        }
    }

    private static int AddSkip(int top, int skip) =>
        top > int.MaxValue - skip ? int.MaxValue : top + skip;

    private static async IAsyncEnumerable<T> ToAsyncEnumerable<T>(
        IEnumerable<T> items, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        foreach (T item in items)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return item;
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    private void ThrowIfDeleted()
    {
        if (_deleted)
        {
            throw new VectorStoreException($"collection '{Name}' has been deleted")
            {
                CollectionName = Name,
            };
        }
    }
}
