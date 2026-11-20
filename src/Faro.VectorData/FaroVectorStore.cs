using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace Faro.VectorData;

/// <summary>
/// A <see cref="VectorStore"/> over a Faro database. Collections map one-to-one onto durable
/// <see cref="FaroDbCollection{TKey,TPayload}"/> instances.
/// </summary>
/// <remarks>
/// Instances are thread-safe. The store owns the underlying <see cref="FaroDb"/> only when created
/// from a path or through <see cref="InMemory(FaroVectorStoreOptions?, FaroOptions?)"/>; when given
/// an existing database, disposing the store leaves that database open.
/// </remarks>
public sealed class FaroVectorStore : VectorStore
{
    private readonly object _gate = new();
    private readonly Dictionary<string, object> _collections = new(StringComparer.Ordinal);
    private readonly FaroDb _db;
    private readonly bool _ownsDb;
    private readonly FaroVectorStoreOptions _options;

    /// <summary>Wraps an existing Faro database.</summary>
    /// <param name="database">The database to use.</param>
    /// <param name="options">Serializer and index configuration.</param>
    /// <param name="ownsDb">Whether disposing this store also disposes <paramref name="database"/>.</param>
    public FaroVectorStore(FaroDb database, FaroVectorStoreOptions? options = null, bool ownsDb = false)
    {
        ArgumentNullException.ThrowIfNull(database);
        _db = database;
        _ownsDb = ownsDb;
        _options = options ?? new FaroVectorStoreOptions();
    }

    /// <summary>Opens (or creates) a Faro database file and exposes it as a vector store.</summary>
    /// <param name="path">Path to the <c>.faro</c> file.</param>
    /// <param name="databaseOptions">Faro durability options.</param>
    /// <param name="options">Serializer and index configuration.</param>
    public FaroVectorStore(
        string path,
        FaroOptions? databaseOptions = null,
        FaroVectorStoreOptions? options = null)
        : this(FaroDb.Open(path, databaseOptions), options, ownsDb: true)
    {
    }

    /// <summary>Creates a vector store over a purely in-memory Faro database.</summary>
    /// <param name="options">Serializer and index configuration.</param>
    /// <param name="databaseOptions">Faro options (durability settings have no effect).</param>
    public static FaroVectorStore InMemory(
        FaroVectorStoreOptions? options = null, FaroOptions? databaseOptions = null) =>
        new(FaroDb.InMemory(databaseOptions), options, ownsDb: true);

    /// <summary>The underlying Faro database.</summary>
    internal FaroDb Database => _db;

    /// <inheritdoc />
    [RequiresDynamicCode("Attribute-based record models are built with reflection; the matching annotations are required by VectorData.Abstractions.")]
    [RequiresUnreferencedCode("Attribute-based record models are built with reflection; use GetDynamicCollection with an explicit definition for trimming.")]
    public override VectorStoreCollection<TKey, TRecord> GetCollection<TKey, TRecord>(
        string name, VectorStoreCollectionDefinition? definition = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        lock (_gate)
        {
            if (_collections.TryGetValue(name, out object? existing))
            {
                return existing as VectorStoreCollection<TKey, TRecord>
                    ?? throw new VectorStoreException($"collection '{name}' is already open with a different schema")
                    {
                        VectorStoreSystemName = "faro",
                        CollectionName = name,
                    };
            }

            Faro.Payload.IPayloadSerializer<TRecord> serializer = ResolveSerializer<TRecord>(name);
            CollectionModel model = BuildModel(
                () => new FaroModelBuilder().Build(typeof(TRecord), typeof(TKey), definition, _options.EmbeddingGenerator),
                name);
            var collection = new FaroVectorStoreCollection<TKey, TRecord>(this, _db, name, model, serializer, _options);
            _collections.Add(name, collection);
            return collection;
        }
    }

    /// <inheritdoc />
    public override VectorStoreCollection<object, Dictionary<string, object?>> GetDynamicCollection(
        string name, VectorStoreCollectionDefinition definition)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(definition);
        lock (_gate)
        {
            if (_collections.TryGetValue(name, out object? existing))
            {
                return existing as VectorStoreCollection<object, Dictionary<string, object?>>
                    ?? throw new VectorStoreException($"collection '{name}' is already open with a different schema")
                    {
                        VectorStoreSystemName = "faro",
                        CollectionName = name,
                    };
            }

            Faro.Payload.IPayloadSerializer<Dictionary<string, object?>> serializer =
                ResolveSerializer<Dictionary<string, object?>>(name);
            CollectionModel model = BuildModel(
                () => new FaroModelBuilder().BuildDynamic(definition, _options.EmbeddingGenerator),
                name);
            var collection = new FaroVectorStoreCollection<object, Dictionary<string, object?>>(
                this, _db, name, model, serializer, _options);
            _collections.Add(name, collection);
            return collection;
        }
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<string> ListCollectionNamesAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        foreach (string name in _db.CollectionNames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return name;
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <inheritdoc />
    public override Task<bool> CollectionExistsAsync(
        string name, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(_db.CollectionNames.Contains(name));
    }

    /// <inheritdoc />
    public override Task EnsureCollectionDeletedAsync(
        string name, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        cancellationToken.ThrowIfCancellationRequested();
        _db.DeleteCollection(name);
        lock (_gate)
        {
            _collections.Remove(name);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceType == typeof(VectorStoreMetadata))
        {
            return new VectorStoreMetadata
            {
                VectorStoreSystemName = "faro",
                VectorStoreName = _db.Path ?? "in-memory",
            };
        }

        if (serviceType == typeof(FaroDb))
        {
            return _db;
        }

        return null;
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing && _ownsDb)
        {
            _db.Dispose();
        }

        base.Dispose(disposing);
    }

    private Faro.Payload.IPayloadSerializer<TRecord> ResolveSerializer<TRecord>(string name)
        where TRecord : class =>
        _options.ResolveSerializer<TRecord>() ?? throw new VectorStoreException(
            $"no source-generated payload serializer is registered for record type '{typeof(TRecord)}' " +
            "(set FaroVectorStoreOptions.JsonSerializerContext or call RegisterPayloadSerializer)")
        {
            VectorStoreSystemName = "faro",
            CollectionName = name,
        };

    private static CollectionModel BuildModel(Func<CollectionModel> build, string name)
    {
        try
        {
            return build();
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException or NotSupportedException)
        {
            throw new VectorStoreException($"invalid record definition for collection '{name}': {exception.Message}", exception)
            {
                VectorStoreSystemName = "faro",
                CollectionName = name,
            };
        }
    }
}
