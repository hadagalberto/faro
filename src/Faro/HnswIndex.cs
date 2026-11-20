using System.Buffers;
using System.Runtime.InteropServices;
using Faro.Internal;
using Faro.Persistence;

namespace Faro;

/// <summary>
/// Hierarchical Navigable Small World graph index (Malkov &amp; Yashunin, 2016/2018).
/// </summary>
/// <remarks>
/// <para>
/// Nodes are assigned an exponentially distributed level and linked on every layer from their
/// level down to 0. Searches enter at the highest node and greedily descend, then run a
/// best-first beam search on layer 0. Construction uses Algorithm 4's heuristic neighbour
/// selection so the graph stays diverse rather than degenerating into a nearest-neighbour chain.
/// </para>
/// <para>
/// Concurrency: each node has its own gate; inserts only lock the few nodes they link, so many
/// inserts proceed in parallel. Adjacency lists are copy-on-write, so searches traverse them
/// without locks and run concurrently with inserts and deletes. Topology snapshots are captured
/// under a short state lock; the expensive distance work happens outside it. Deletes are
/// tombstones with neighbour repair, and <see cref="Compact"/> rebuilds a graph free of deleted
/// nodes while preserving stable ids.
/// </para>
/// <para>
/// The <see cref="Search(ReadOnlySpan{float},int,Span{SearchResult})"/> overload is allocation-free
/// in steady state: heaps, visited stamps and candidate buffers are pooled.
/// </para>
/// </remarks>
public sealed class HnswIndex
{
    // Levels beyond this are astronomically unlikely for the paper's mL = 1/ln(M); the cap keeps
    // per-node layer arrays bounded even for adversarial seeds and small M.
    private const int MaxSupportedLevel = 24;

    private readonly int _dimensions;
    private readonly Distance _distance;
    private readonly bool _normalize;
    private readonly bool _lowerIsCloser;
    private readonly int _wordsPerVector;
    private readonly int _m;
    private readonly int _m0;
    private readonly int _efConstruction;
    private readonly int _efSearch;
    private readonly double _levelMultiplier;
    private readonly ulong? _seed;
    private readonly bool _extendCandidates;
    private readonly bool _keepPruned;
    private readonly double _mc;

    private readonly QuantizationOptions? _quantization;
    private readonly int _codeLength;
    private readonly object _quantLock = new();
    private readonly List<float[]> _trainingSample = new();
    private byte[]? _codes;
    private Quantizer? _quantizer;

    private readonly object _stateLock = new();
    private readonly object _deleteLock = new();
    private HnswNode?[] _nodes;
    private int _count;
    private int _entryPointId = -1;
    private int _maxLevel = -1;
    private int _liveCount;

    /// <summary>Creates an HNSW index with the given options.</summary>
    /// <param name="options">The index configuration.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is outside its valid range.</exception>
    /// <exception cref="ArgumentException"><see cref="HnswOptions.NormalizeOnAdd"/> is set for a non-cosine metric.</exception>
    public HnswIndex(HnswOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        if (options.Dimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.Dimensions, "Dimensions must be positive.");
        }

        if (options.InitialCapacity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.InitialCapacity, "InitialCapacity must be positive.");
        }

        if (options.M < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.M, "M must be at least 2.");
        }

        if (options.EfConstruction < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.EfConstruction, "EfConstruction must be positive.");
        }

        if (options.EfSearch < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.EfSearch, "EfSearch must be positive.");
        }

        if (options.Mc < 1.0 || double.IsNaN(options.Mc) || double.IsInfinity(options.Mc))
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.Mc, "Mc must be a finite value >= 1.");
        }

        double multiplier = options.LevelMultiplier ?? (1.0 / Math.Log(options.M));
        if (multiplier <= 0 || double.IsNaN(multiplier) || double.IsInfinity(multiplier))
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.LevelMultiplier, "LevelMultiplier must be a finite positive value.");
        }

        _dimensions = options.Dimensions;
        _distance = options.Distance;
        _normalize = options.NormalizeOnAdd ?? (options.Distance == Distance.Cosine);
        if (_normalize && _distance != Distance.Cosine)
        {
            throw new ArgumentException("NormalizeOnAdd is only supported for Distance.Cosine.", nameof(options));
        }

        _lowerIsCloser = Distances.LowerIsCloser(_distance);
        _wordsPerVector = BitVectors.WordsFor(_dimensions);
        _m = options.M;
        _m0 = 2 * options.M;
        _efConstruction = options.EfConstruction;
        _efSearch = options.EfSearch;
        _levelMultiplier = multiplier;
        _seed = options.Seed is null ? null : unchecked((ulong)options.Seed.Value);
        _extendCandidates = options.ExtendCandidates;
        _keepPruned = options.KeepPrunedConnections;
        _mc = options.Mc;
        _nodes = new HnswNode?[options.InitialCapacity];

        _quantization = options.Quantization;
        if (_quantization is not null)
        {
            _quantization.ValidateCompatibility(_dimensions, _distance);
            _codeLength = _quantization.CodeLengthFor(_dimensions, _distance);
            if ((long)options.InitialCapacity * _codeLength > Array.MaxLength)
            {
                throw new ArgumentOutOfRangeException(nameof(options), options.InitialCapacity, "InitialCapacity exceeds the maximum code array size.");
            }

            _codes = new byte[options.InitialCapacity * _codeLength];
            _quantizer = _quantization.PreTrained;
        }
    }

    /// <summary>Total number of nodes inserted, including tombstoned ones.</summary>
    public int Count
    {
        get { lock (_stateLock) { return _count; } }
    }

    /// <summary>Number of live (non-deleted) nodes.</summary>
    public int LiveCount => Volatile.Read(ref _liveCount);

    /// <summary>Number of dimensions of stored vectors.</summary>
    public int Dimensions => _dimensions;

    /// <summary>The distance metric of this index.</summary>
    public Distance Distance => _distance;

    /// <summary>Whether vectors are normalized on insertion (relevant for cosine scoring).</summary>
    internal bool NormalizesOnAdd => _normalize;

    /// <summary>The configured default search beam.</summary>
    internal int DefaultEfSearch => _efSearch;

    /// <summary>Highest layer currently present in the graph (-1 when empty).</summary>
    public int MaxLevel
    {
        get { lock (_stateLock) { return _maxLevel; } }
    }

    /// <summary>Id of the current entry point (-1 when empty). Internal for invariant tests.</summary>
    internal int EntryPointId
    {
        get { lock (_stateLock) { return _entryPointId; } }
    }

    /// <summary>Captures a consistent topology snapshot. Internal for invariant tests.</summary>
    internal HnswGraphView CurrentView => CaptureView();

    /// <summary>
    /// Inserts a vector and returns its stable internal id (sequential from 0). Ids are never
    /// reused, including after deletes and <see cref="Compact"/>.
    /// </summary>
    /// <param name="vector">The vector to store; its length must equal <see cref="Dimensions"/>.</param>
    /// <returns>The internal id of the inserted node.</returns>
    /// <exception cref="ArgumentException"><paramref name="vector"/> has the wrong length or a non-finite value.</exception>
    /// <remarks>Many inserts may run concurrently; each only locks the nodes it links.</remarks>
    public int Add(ReadOnlySpan<float> vector)
    {
        ValidateVector(vector);

        float[] stored;
        ulong[]? bits = null;
        if (_distance == Distance.Hamming)
        {
            bits = new ulong[_wordsPerVector];
            BitVectors.PackSignBits(vector, bits);
            stored = [];
        }
        else
        {
            stored = new float[_dimensions];
            if (_normalize)
            {
                Distances.CopyNormalized(vector, stored);
            }
            else
            {
                vector.CopyTo(stored);
            }
        }

        int id;
        int oldEntry;
        int oldMax;
        HnswNode node;
        HnswGraphView view;
        lock (_stateLock)
        {
            id = _count;
            oldEntry = _entryPointId;
            oldMax = _maxLevel;

            int level = AssignLevel(id);
            EnsureCapacity(id + 1);
            node = new HnswNode(
                id,
                level,
                VectorData.FromFloats(stored),
                bits is null ? null : VectorData.FromBits(bits),
                level);
            _nodes[id] = node;
            _count = id + 1;
            if (_maxLevel < level)
            {
                _maxLevel = level;
                _entryPointId = id;
            }

            view = new HnswGraphView(_nodes, _count, _entryPointId, _maxLevel);
        }

        if (_quantization is not null)
        {
            ReadOnlySpan<float> encodeSource = _distance == Distance.Hamming ? vector : node.Vector.Floats;
            lock (_quantLock)
            {
                Quantizer? quantizer = Volatile.Read(ref _quantizer);
                if (quantizer is null)
                {
                    if (_trainingSample.Count < QuantizationRuntime.TrainingSampleSize)
                    {
                        _trainingSample.Add(encodeSource.ToArray());
                    }
                }
                else if (_distance == Distance.Hamming)
                {
                    quantizer.EncodePacked(node.Bits!.Bits, _codes!.AsSpan(id * _codeLength, _codeLength));
                }
                else
                {
                    quantizer.Encode(encodeSource, _codes!.AsSpan(id * _codeLength, _codeLength));
                }
            }
        }

        InsertNode(node, view, oldEntry, oldMax);
        Interlocked.Increment(ref _liveCount);
        return id;
    }

    /// <summary>Searches for the <paramref name="k"/> nearest live nodes using the configured efSearch.</summary>
    /// <param name="query">The query vector; its length must equal <see cref="Dimensions"/>.</param>
    /// <param name="k">Maximum number of hits; must be positive.</param>
    /// <returns>Best-first hits in a freshly allocated array of at most min(k, live count).</returns>
    public SearchResult[] Search(ReadOnlySpan<float> query, int k) => Search(query, k, _efSearch);

    /// <summary>Searches for the <paramref name="k"/> nearest live nodes with an explicit beam width.</summary>
    /// <param name="query">The query vector.</param>
    /// <param name="k">Maximum number of hits; must be positive.</param>
    /// <param name="efSearch">Beam width; at least <paramref name="k"/> is used.</param>
    /// <returns>Best-first hits in a freshly allocated array.</returns>
    public SearchResult[] Search(ReadOnlySpan<float> query, int k, int efSearch)
    {
        ValidateEf(efSearch);
        int capacity = EstimateResultCapacity(k);
        var results = new SearchResult[capacity];
        int written = SearchCore(query, k, results, efSearch);
        return written == results.Length ? results : results.AsSpan(0, written).ToArray();
    }

    /// <summary>
    /// Searches into <paramref name="results"/> with the configured efSearch. Allocation-free in
    /// steady state; returns the number of hits written (at most min(k, results.Length, live count)).
    /// </summary>
    /// <param name="query">The query vector.</param>
    /// <param name="k">Maximum number of hits; must be positive.</param>
    /// <param name="results">Destination buffer.</param>
    public int Search(ReadOnlySpan<float> query, int k, Span<SearchResult> results) =>
        Search(query, k, results, _efSearch);

    /// <summary>Searches into <paramref name="results"/> with an explicit beam width.</summary>
    /// <param name="query">The query vector.</param>
    /// <param name="k">Maximum number of hits; must be positive.</param>
    /// <param name="results">Destination buffer.</param>
    /// <param name="efSearch">Beam width; at least <paramref name="k"/> is used.</param>
    public int Search(ReadOnlySpan<float> query, int k, Span<SearchResult> results, int efSearch)
    {
        ValidateEf(efSearch);
        return SearchCore(query, k, results, efSearch);
    }

    /// <summary>
    /// Marks a node deleted (tombstone), removes all links pointing to it and reconnects its
    /// neighbours so the live graph stays traversable. The node id is never reused.
    /// </summary>
    /// <param name="id">The internal id to delete.</param>
    /// <returns><see langword="true"/> when a live node was deleted; <see langword="false"/> when the id is unknown or already deleted.</returns>
    public bool Delete(int id)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(id);

        lock (_deleteLock)
        {
            HnswGraphView view = CaptureView();
            if (id >= view.Count)
            {
                return false;
            }

            HnswNode? node = view.Nodes[id];
            if (node is null || node.Deleted)
            {
                return false;
            }

            node.Deleted = true;
            Interlocked.Decrement(ref _liveCount);

            for (int layer = 0; layer <= node.Level; layer++)
            {
                // Pruning can leave links asymmetric, so sweep every node: a live node may hold
                // a link to the victim even when the victim no longer links back to it.
                var repair = new List<int>();
                for (int otherId = 0; otherId < view.Count; otherId++)
                {
                    if (otherId == id)
                    {
                        continue;
                    }

                    HnswNode? other = view.Nodes[otherId];
                    if (other is null || layer > other.Level)
                    {
                        continue;
                    }

                    if (Array.IndexOf(other.GetLinks(layer), id) >= 0)
                    {
                        other.RemoveLinks(layer, id);
                        if (!other.Deleted)
                        {
                            repair.Add(otherId);
                        }
                    }
                }

                // Repair the connectivity spine: drop every reference to the tombstone and
                // re-link the nodes it used to connect so the spanning structure stays whole.
                var spineAffected = new List<int>();
                foreach (int spineId in node.GetBackbone(layer))
                {
                    if ((uint)spineId < (uint)view.Count && view.Nodes[spineId] is { Deleted: false })
                    {
                        spineAffected.Add(spineId);
                    }
                }

                for (int otherId = 0; otherId < view.Count; otherId++)
                {
                    if (otherId == id)
                    {
                        continue;
                    }

                    HnswNode? other = view.Nodes[otherId];
                    if (other is null || layer > other.Level)
                    {
                        continue;
                    }

                    if (Array.IndexOf(other.GetBackbone(layer), id) >= 0)
                    {
                        other.RemoveBackbone(layer, id);
                        if (!other.Deleted)
                        {
                            spineAffected.Add(otherId);
                        }
                    }
                }

                if (spineAffected.Count >= 2)
                {
                    HnswNode? hub = view.Nodes[spineAffected[0]];
                    if (hub is not null)
                    {
                        for (int i = 1; i < spineAffected.Count; i++)
                        {
                            HnswNode? leaf = view.Nodes[spineAffected[i]];
                            if (leaf is not null && !leaf.Deleted)
                            {
                                AddBackboneEdge(hub, leaf, layer, view);
                            }
                        }
                    }
                }

                if (repair.Count < 2)
                {
                    continue;
                }

                int maxDegree = layer == 0 ? _m0 : _m;
                foreach (int a in repair)
                {
                    HnswNode? aNode = view.Nodes[a];
                    if (aNode is null || aNode.Deleted)
                    {
                        continue;
                    }

                    // Simple distance-ranked repair: much cheaper than re-running the full
                    // Algorithm 4 heuristic for every survivor of a tombstone.
                    int[] chosen = SelectClosest(aNode, repair, layer, maxDegree, view);
                    foreach (int b in chosen)
                    {
                        HnswNode? bNode = view.Nodes[b];
                        if (bNode is not null && !bNode.Deleted && b != a)
                        {
                            LinkWithDegreeCap(aNode, bNode, layer, maxDegree, view);
                        }
                    }
                }
            }

            return true;
        }
    }

    /// <summary>
    /// Rebuilds the graph from the live nodes, dropping every tombstone and every link to one.
    /// Node ids and levels are preserved. Searches that started before the call keep using the
    /// previous snapshot; new searches see the compacted graph.
    /// </summary>
    public void Compact()
    {
        lock (_deleteLock)
        {
            lock (_stateLock)
            {
                HnswGraphView oldView = new(_nodes, _count, _entryPointId, _maxLevel);
                var newNodes = new HnswNode?[_nodes.Length];
                int live = 0;

                for (int id = 0; id < oldView.Count; id++)
                {
                    HnswNode? old = oldView.Nodes[id];
                    if (old is null || old.Deleted)
                    {
                        continue;
                    }

                    // Vector descriptors are immutable (updates swap the descriptor), so they can
                    // be shared with the old snapshot — including mapped checkpoint slices.
                    newNodes[id] = new HnswNode(id, old.Level, old.Vector, old.Bits, old.Level);
                    live++;
                }

                int maxLevel = -1;
                int entry = -1;
                for (int id = 0; id < oldView.Count; id++)
                {
                    HnswNode? node = newNodes[id];
                    if (node is null)
                    {
                        continue;
                    }

                    int previousEntry = entry;
                    int previousMax = maxLevel;
                    if (maxLevel < node.Level)
                    {
                        maxLevel = node.Level;
                        entry = id;
                    }

                    var partial = new HnswGraphView(newNodes, id + 1, entry, maxLevel);
                    InsertNode(node, partial, previousEntry, previousMax);
                }

                Volatile.Write(ref _liveCount, live);
                _nodes = newNodes;
                _count = oldView.Count;
                _entryPointId = entry;
                _maxLevel = maxLevel;
            }
        }
    }

    /// <summary>
    /// Copies the stored form of the vector with the given id into <paramref name="destination"/>
    /// (normalized for cosine-with-normalization, sign-reconstructed ±1 values for Hamming).
    /// Deleted nodes still expose their vector until compaction.
    /// </summary>
    /// <param name="id">The internal node id.</param>
    /// <param name="destination">Destination span of <see cref="Dimensions"/> floats.</param>
    /// <returns><see langword="false"/> if the id is unknown.</returns>
    public bool TryGetVector(int id, Span<float> destination)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(id);
        if (destination.Length != _dimensions)
        {
            throw new ArgumentException($"destination must have {_dimensions} entries, got {destination.Length}.", nameof(destination));
        }

        HnswGraphView view = CaptureView();
        if (id >= view.Count)
        {
            return false;
        }

        HnswNode? node = view.Nodes[id];
        if (node is null)
        {
            return false;
        }

        if (_distance == Distance.Hamming)
        {
            BitVectors.UnpackSignBits(node.Bits!.Bits, _dimensions, destination);
        }
        else
        {
            node.Vector.Floats.CopyTo(destination);
        }

        return true;
    }

    // -------------------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------------------

    private void InsertNode(HnswNode node, HnswGraphView view, int entryId, int maxLevelBefore)
    {
        if (entryId < 0 || entryId == node.Id)
        {
            return; // first node in the graph (or sole node at a new top layer)
        }

        var scorer = new QueryScorer(this, node.Vector.Floats, (node.Bits is null ? default : node.Bits.Bits));
        int[] buffer = ArrayPool<int>.Shared.Rent(Math.Max(_efConstruction, 1));
        try
        {
            int entry = entryId;
            for (int layer = maxLevelBefore; layer > node.Level; layer--)
            {
                int found = SearchLayer(view, scorer, entry, 1, layer, buffer.AsSpan(0, 1));
                if (found > 0)
                {
                    entry = buffer[0];
                }
            }

            int firstLayer = Math.Min(node.Level, maxLevelBefore);
            for (int layer = firstLayer; layer >= 0; layer--)
            {
                int found = SearchLayer(view, scorer, entry, _efConstruction, layer, buffer);
                int maxDegree = layer == 0 ? _m0 : _m;
                int[] selected = SelectNeighbors(
                    node, buffer.AsSpan(0, found), layer, maxDegree, scorer, view, extend: _extendCandidates);

                foreach (int neighborId in selected)
                {
                    HnswNode? neighbor = view.Nodes[neighborId];
                    if (neighbor is null || neighbor.Deleted)
                    {
                        continue;
                    }

                    Connect(node, neighbor, layer, maxDegree, view);
                }

                // The nearest selected neighbour becomes this node's spine edge on this layer.
                // It is reciprocal and immune to Algorithm 4 pruning, keeping the layer connected.
                if (selected.Length > 0)
                {
                    HnswNode? parent = view.Nodes[selected[0]];
                    if (parent is not null && !parent.Deleted)
                    {
                        AddBackboneEdge(node, parent, layer, view);
                    }
                }

                if (found > 0)
                {
                    entry = buffer[0];
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    /// <summary>Links two nodes bidirectionally and prunes the neighbour when over its degree bound.</summary>
    private void Connect(HnswNode a, HnswNode b, int layer, int maxDegree, HnswGraphView view)
    {
        int lo = Math.Min(a.Id, b.Id);
        int hi = Math.Max(a.Id, b.Id);
        HnswNode? loNode = view.Nodes[lo];
        HnswNode? hiNode = view.Nodes[hi];
        if (loNode is null || hiNode is null)
        {
            return;
        }

        // Consistent lock order prevents deadlock between concurrent inserters.
        lock (loNode.Gate)
        {
            lock (hiNode.Gate)
            {
                a.AddLink(layer, b.Id);
                b.AddLink(layer, a.Id);

                int[] links = b.GetLinks(layer);
                if (links.Length > maxDegree)
                {
                    // Pruning an existing node only considers its own links: re-expanding every
                    // candidate's neighbourhood here would make each insert O(M * ef * degree).
                    int[] pruned = SelectNeighbors(
                        b, links, layer, maxDegree,
                        new QueryScorer(this, b.Vector.Floats, b.Bits is null ? default : b.Bits.Bits), view, extend: false, protectedId: a.Id);
                    b.SetLinks(layer, pruned);
                }
            }
        }
    }

    /// <summary>Returns the closest live candidates to a node, used by the cheap delete repair.</summary>
    private int[] SelectClosest(HnswNode query, IReadOnlyList<int> candidates, int layer, int limit, HnswGraphView view)
    {
        var scorer = new QueryScorer(this, query.Vector.Floats, query.Bits is null ? default : query.Bits.Bits);
        var scored = new List<(float Score, int Id)>(candidates.Count);
        for (int i = 0; i < candidates.Count; i++)
        {
            int id = candidates[i];
            // A concurrent insert may have added a link to an id newer than this snapshot.
            if (id == query.Id || (uint)id >= (uint)view.Count
                || view.Nodes[id] is not { Deleted: false })
            {
                continue;
            }

            scored.Add((scorer.Score(view.Nodes[id]!), id));
        }

        scored.Sort((x, y) =>
        {
            int byScore = x.Item1.CompareTo(y.Item1);
            if (!_lowerIsCloser)
            {
                byScore = -byScore;
            }

            return byScore != 0 ? byScore : x.Item2.CompareTo(y.Item2);
        });

        int count = Math.Min(limit, scored.Count);
        var result = new int[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = scored[i].Item2;
        }

        return result;
    }

    /// <summary>Reciprocal repair link capped to the Algorithm 4 degree bound.</summary>
    private void LinkWithDegreeCap(HnswNode a, HnswNode b, int layer, int maxDegree, HnswGraphView view)
    {
        int lo = Math.Min(a.Id, b.Id);
        int hi = Math.Max(a.Id, b.Id);
        HnswNode? loNode = view.Nodes[lo];
        HnswNode? hiNode = view.Nodes[hi];
        if (loNode is null || hiNode is null)
        {
            return;
        }

        lock (loNode.Gate)
        {
            lock (hiNode.Gate)
            {
                a.AddLink(layer, b.Id);
                b.AddLink(layer, a.Id);
                TrimLinks(a, layer, maxDegree, view);
                TrimLinks(b, layer, maxDegree, view);
            }
        }
    }

    private void TrimLinks(HnswNode node, int layer, int maxDegree, HnswGraphView view)
    {
        int[] links = node.GetLinks(layer);
        if (links.Length <= maxDegree)
        {
            return;
        }

        node.SetLinks(layer, SelectClosest(node, links, layer, maxDegree, view));
    }

    /// <summary>Adds a reciprocal spine edge. Lock order by id keeps concurrent inserts deadlock-free.</summary>
    private static void AddBackboneEdge(HnswNode a, HnswNode b, int layer, HnswGraphView view)
    {
        int lo = Math.Min(a.Id, b.Id);
        int hi = Math.Max(a.Id, b.Id);
        HnswNode? loNode = view.Nodes[lo];
        HnswNode? hiNode = view.Nodes[hi];
        if (loNode is null || hiNode is null)
        {
            return;
        }

        lock (loNode.Gate)
        {
            lock (hiNode.Gate)
            {
                a.AddBackbone(layer, b.Id);
                b.AddBackbone(layer, a.Id);
            }
        }
    }

    /// <summary>
    /// Algorithm 4. Returns at most <paramref name="maxDegree"/> diverse neighbours of
    /// <paramref name="query"/> chosen from <paramref name="candidates"/>, with the optional
    /// candidate extension and pruned-connection fallback from the options.
    /// </summary>
    private int[] SelectNeighbors(
        HnswNode query,
        ReadOnlySpan<int> candidates,
        int layer,
        int maxDegree,
        QueryScorer scorer,
        HnswGraphView view,
        bool extend = true,
        int protectedId = -1)
    {
        var working = new List<int>(candidates.Length);
        var seen = new HashSet<int>();
        for (int i = 0; i < candidates.Length; i++)
        {
            AddCandidate(candidates[i], query.Id, view, seen, working);
        }

        if (extend)
        {
            int initial = working.Count;
            for (int i = 0; i < initial; i++)
            {
                HnswNode? node = view.Nodes[working[i]];
                if (node is null)
                {
                    continue;
                }

                int[] adjacent = node.GetLinks(layer);
                for (int j = 0; j < adjacent.Length; j++)
                {
                    AddCandidate(adjacent[j], query.Id, view, seen, working);
                }
            }
        }

        // Score and sort once (as in hnswlib), then walk the ranked list applying the diversity
        // test. Re-scanning the whole working set for each of M slots is quadratic and dominates
        // build time once extendCandidates has grown the candidate set.
        var ranked = new List<(float Score, int Id)>(working.Count);
        foreach (int id in working)
        {
            ranked.Add((ScoreId(id, scorer, view), id));
        }

        ranked.Sort(CompareByMetric);

        var selected = new List<int>(maxDegree);
        var pruned = new List<(float Score, int Id)>();
        for (int i = 0; i < ranked.Count; i++)
        {
            (float score, int id) = ranked[i];
            bool keep = true;
            if (selected.Count > 0)
            {
                HnswNode chosenNode = view.Nodes[id]!;
                for (int j = 0; j < selected.Count; j++)
                {
                    HnswNode? other = view.Nodes[selected[j]];
                    if (other is not null && !Diverse(score, ScoreNodeToNode(chosenNode, other)))
                    {
                        keep = false;
                        break;
                    }
                }
            }

            if (keep && selected.Count < maxDegree)
            {
                selected.Add(id);
            }
            else
            {
                pruned.Add((score, id));
            }
        }

        if (_keepPruned && selected.Count < maxDegree)
        {
            int slots = maxDegree - selected.Count;
            for (int i = 0; i < slots && i < pruned.Count; i++)
            {
                selected.Add(pruned[i].Id);
            }
        }

        // The just-linked node must stay reachable from the entry point. Without this guard,
        // pruning an older neighbour could drop the only reverse link to the new node and
        // strand it in the graph (its own outgoing links cannot be used to reach it).
        if (protectedId >= 0 && seen.Contains(protectedId) && !selected.Contains(protectedId))
        {
            if (selected.Count < maxDegree)
            {
                selected.Add(protectedId);
            }
            else
            {
                selected[selected.Count - 1] = protectedId;
            }
        }

        return selected.ToArray();
    }

    private static void AddCandidate(int id, int excluded, HnswGraphView view, HashSet<int> seen, List<int> result)
    {
        if (id == excluded || (uint)id >= (uint)view.Count
            || view.Nodes[id] is not { Deleted: false })
        {
            return;
        }

        if (seen.Add(id))
        {
            result.Add(id);
        }
    }

    // -------------------------------------------------------------------------------------
    // Search
    // -------------------------------------------------------------------------------------

    /// <summary>Algorithm 5: best-first layer search. Fills <paramref name="destination"/> best-first.</summary>
    private int SearchLayer(
        HnswGraphView view,
        QueryScorer scorer,
        int entryId,
        int ef,
        int layer,
        Span<int> destination)
    {
        (HeapOrder bestFirst, HeapOrder worstKept) = Heaps;
        using var candidates = DistanceHeap.Rent(bestFirst, ef);
        using var top = DistanceHeap.Rent(worstKept, ef);
        using var visited = VisitedSet.Rent(Math.Max(view.Count, 1));

        float entryScore = ScoreId(entryId, scorer, view);
        candidates.Push(entryScore, entryId);
        top.PushBounded(ef, entryScore, entryId);
        visited.Add(entryId);

        while (candidates.Count > 0)
        {
            candidates.TryPeekRoot(out float candidateScore, out _);
            top.TryPeekRoot(out float worstScore, out _);
            if (top.Count == ef && !Closer(candidateScore, worstScore))
            {
                break;
            }

            candidates.TryPop(out candidateScore, out int currentId);
            HnswNode? current = view.Nodes[currentId];
            if (current is null)
            {
                continue;
            }

            // Traverse both the pruned Algorithm 4 links and the unpruned connectivity spine.
            for (int source = 0; source < 2; source++)
            {
                int[] adjacent = source == 0 ? current.GetLinks(layer) : current.GetBackbone(layer);
                for (int i = 0; i < adjacent.Length; i++)
                {
                    int id = adjacent[i];
                    if ((uint)id >= (uint)view.Count || !visited.Add(id))
                    {
                        continue;
                    }

                    HnswNode? neighbor = view.Nodes[id];
                    if (neighbor is null || neighbor.Deleted)
                    {
                        continue;
                    }

                    float score = scorer.Score(neighbor);
                    if (top.Count < ef || Closer(score, worstScore))
                    {
                        candidates.Push(score, id);
                        top.PushBounded(ef, score, id);
                        top.TryPeekRoot(out worstScore, out _);
                    }
                }
            }
        }

        int count = top.Count;
        int index = count;
        while (top.TryPop(out _, out int id))
        {
            destination[--index] = id;
        }

        return count;
    }

    /// <summary>Trains the codec (when lazy) and encodes every node exactly once, on first search.</summary>
    private void EnsureQuantized()
    {
        QuantizationOptions options = _quantization!;
        if (Volatile.Read(ref _quantizer) is not null)
        {
            return;
        }

        lock (_quantLock)
        {
            if (Volatile.Read(ref _quantizer) is not null)
            {
                return;
            }

            Quantizer quantizer = QuantizationRuntime.Train(options, _trainingSample, _dimensions, _distance);
            HnswGraphView view = CaptureView();
            for (int id = 0; id < view.Count; id++)
            {
                HnswNode? node = view.Nodes[id];
                if (node is null)
                {
                    continue;
                }

                Span<byte> code = _codes!.AsSpan(id * _codeLength, _codeLength);
                if (_distance == Distance.Hamming)
                {
                    quantizer.EncodePacked(node.Bits!.Bits, code);
                }
                else
                {
                    quantizer.Encode(node.Vector.Floats, code);
                }
            }

            _trainingSample.Clear();
            Volatile.Write(ref _quantizer, quantizer);
        }
    }

    private float ScoreNodeQuantized(ReadOnlySpan<byte> queryCode, ReadOnlySpan<byte> codes, HnswNode node) =>
        Volatile.Read(ref _quantizer)!.Distance(queryCode, codes.Slice(node.Id * _codeLength, _codeLength));

    private int SearchCore(ReadOnlySpan<float> query, int k, Span<SearchResult> results, int efSearch)
    {
        ValidateQuery(query);
        ValidateK(k);

        HnswGraphView view = CaptureView();
        if (view.EntryPointId < 0 || view.Count == 0 || results.Length == 0)
        {
            return 0;
        }

        int limit = Math.Min(k, results.Length);
        int beam = Math.Max(efSearch, limit);
        bool quantized = false;
        if (_quantization is not null)
        {
            EnsureQuantized();
            quantized = Volatile.Read(ref _quantizer) is not null;
            if (quantized)
            {
                // Oversampling: widen the beam so the rescoring pass sees k * factor candidates.
                beam = Math.Max(beam, (int)Math.Min(view.Count, (long)limit * _quantization.RescoreFactor));
            }
        }

        float[]? normalized = null;
        ReadOnlySpan<float> effective = query;
        ulong[]? packed = null;
        if (_normalize)
        {
            normalized = ArrayPool<float>.Shared.Rent(_dimensions);
            Distances.CopyNormalized(query, normalized.AsSpan(0, _dimensions));
            effective = normalized.AsSpan(0, _dimensions);
        }

        if (_distance == Distance.Hamming)
        {
            packed = ArrayPool<ulong>.Shared.Rent(_wordsPerVector);
            BitVectors.PackSignBits(query, packed.AsSpan(0, _wordsPerVector));
        }

        byte[]? queryCode = null;
        if (quantized)
        {
            queryCode = ArrayPool<byte>.Shared.Rent(_codeLength);
            Span<byte> querySpan = queryCode.AsSpan(0, _codeLength);
            if (_distance == Distance.Hamming)
            {
                Volatile.Read(ref _quantizer)!.EncodePacked(packed!.AsSpan(0, _wordsPerVector), querySpan);
            }
            else
            {
                Volatile.Read(ref _quantizer)!.Encode(effective, querySpan);
            }
        }

        int[] candidates = ArrayPool<int>.Shared.Rent(beam);
        try
        {
            ReadOnlySpan<ulong> packedQuery = packed is null ? default : packed.AsSpan(0, _wordsPerVector);
            QueryScorer scorer;
            if (quantized)
            {
                byte[] codes = _codes!;
                scorer = new QueryScorer(this, queryCode!.AsSpan(0, _codeLength), codes.AsSpan());
            }
            else
            {
                scorer = new QueryScorer(this, effective, packedQuery);
            }

            int entry = view.EntryPointId;
            int found = 0;
            for (int layer = view.MaxLevel; layer >= 0; layer--)
            {
                found = SearchLayer(view, scorer, entry, beam, layer, candidates);
                if (found > 0)
                {
                    entry = candidates[0];
                }
            }

            using var resultHeap = DistanceHeap.Rent(Heaps.WorstKept, limit);
            for (int i = 0; i < found; i++)
            {
                int id = candidates[i];
                HnswNode? node = view.Nodes[id];
                if (node is null || node.Deleted)
                {
                    continue;
                }

                // Quantization only steers traversal; the returned score and ordering always
                // come from the stored full-precision vector.
                float score = quantized
                    ? ScoreQueryNode(effective, packedQuery, node)
                    : scorer.Score(node);
                resultHeap.PushBounded(limit, score, id);
            }

            int written = resultHeap.Count;
            int index = written;
            while (resultHeap.TryPop(out float score, out int id))
            {
                results[--index] = new SearchResult(id, score);
            }

            return written;
        }
        finally
        {
            if (normalized is not null)
            {
                ArrayPool<float>.Shared.Return(normalized);
            }

            if (packed is not null)
            {
                ArrayPool<ulong>.Shared.Return(packed);
            }

            if (queryCode is not null)
            {
                ArrayPool<byte>.Shared.Return(queryCode);
            }

            ArrayPool<int>.Shared.Return(candidates);
        }
    }

    // -------------------------------------------------------------------------------------
    // Filtered search
    // -------------------------------------------------------------------------------------

    /// <summary>
    /// Searches the graph for the <paramref name="k"/> nearest vectors that satisfy
    /// <paramref name="isMatch"/>, using the configured beam width.
    /// </summary>
    /// <param name="query">The query vector.</param>
    /// <param name="k">Maximum number of hits.</param>
    /// <param name="isMatch">Predicate evaluated on internal ids during traversal.</param>
    /// <param name="efSearch">Beam width; at least <paramref name="k"/> is used.</param>
    /// <returns>Best-first hits in a freshly allocated array.</returns>
    public SearchResult[] SearchFiltered(
        ReadOnlySpan<float> query,
        int k,
        Func<int, bool> isMatch,
        int efSearch) =>
        SearchFiltered(query, k, isMatch, efSearch, null);

    /// <summary>
    /// Searches the graph for matching vectors, seeding the beam with ids from the payload index
    /// so sparse filters still reach their matching regions.
    /// </summary>
    /// <param name="query">The query vector.</param>
    /// <param name="k">Maximum number of hits.</param>
    /// <param name="isMatch">Predicate evaluated on internal ids during traversal.</param>
    /// <param name="efSearch">Beam width; at least <paramref name="k"/> is used.</param>
    /// <param name="seedIds">
    /// Ids known to satisfy the filter from the payload indexes. They are inserted into the beam
    /// (at most <paramref name="efSearch"/> of them) to jump directly into matching regions.
    /// </param>
    /// <returns>Best-first hits in a freshly allocated array.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="isMatch"/> is <see langword="null"/>.</exception>
    public SearchResult[] SearchFiltered(
        ReadOnlySpan<float> query,
        int k,
        Func<int, bool> isMatch,
        int efSearch,
        IReadOnlyCollection<int>? seedIds)
    {
        ArgumentNullException.ThrowIfNull(isMatch);
        ValidateEf(efSearch);
        int capacity = EstimateResultCapacity(k);
        if (capacity == 0)
        {
            return [];
        }

        var results = new SearchResult[capacity];
        int written = SearchFilteredCore(query, k, results, efSearch, isMatch, seedIds);
        return written == results.Length ? results : results.AsSpan(0, written).ToArray();
    }

    private int SearchFilteredCore(
        ReadOnlySpan<float> query,
        int k,
        Span<SearchResult> results,
        int efSearch,
        Func<int, bool> isMatch,
        IReadOnlyCollection<int>? seedIds)
    {
        ValidateQuery(query);
        ValidateK(k);

        HnswGraphView view = CaptureView();
        if (view.EntryPointId < 0 || view.Count == 0 || results.Length == 0)
        {
            return 0;
        }

        int limit = Math.Min(k, results.Length);
        int beam = Math.Max(efSearch, limit);
        bool quantized = false;
        if (_quantization is not null)
        {
            EnsureQuantized();
            quantized = Volatile.Read(ref _quantizer) is not null;
            if (quantized)
            {
                beam = Math.Max(beam, (int)Math.Min(view.Count, (long)limit * _quantization.RescoreFactor));
            }
        }

        float[]? normalized = null;
        ReadOnlySpan<float> effective = query;
        ulong[]? packed = null;
        if (_normalize)
        {
            normalized = ArrayPool<float>.Shared.Rent(_dimensions);
            Distances.CopyNormalized(query, normalized.AsSpan(0, _dimensions));
            effective = normalized.AsSpan(0, _dimensions);
        }

        if (_distance == Distance.Hamming)
        {
            packed = ArrayPool<ulong>.Shared.Rent(_wordsPerVector);
            BitVectors.PackSignBits(query, packed.AsSpan(0, _wordsPerVector));
        }

        byte[]? queryCode = null;
        if (quantized)
        {
            queryCode = ArrayPool<byte>.Shared.Rent(_codeLength);
            Span<byte> querySpan = queryCode.AsSpan(0, _codeLength);
            if (_distance == Distance.Hamming)
            {
                Volatile.Read(ref _quantizer)!.EncodePacked(packed!.AsSpan(0, _wordsPerVector), querySpan);
            }
            else
            {
                Volatile.Read(ref _quantizer)!.Encode(effective, querySpan);
            }
        }

        int[] candidates = ArrayPool<int>.Shared.Rent(beam);
        try
        {
            ReadOnlySpan<ulong> packedQuery = packed is null ? default : packed.AsSpan(0, _wordsPerVector);
            QueryScorer scorer;
            if (quantized)
            {
                byte[] codes = _codes!;
                scorer = new QueryScorer(this, queryCode!.AsSpan(0, _codeLength), codes.AsSpan());
            }
            else
            {
                scorer = new QueryScorer(this, effective, packedQuery);
            }

            int entry = view.EntryPointId;
            int found = 0;

            // Upper layers are traversed without the filter: they only choose a good entry point
            // for the layer-0 filtered beam, and filtering them could strand the search far from
            // any matching region.
            for (int layer = view.MaxLevel; layer > 0; layer--)
            {
                found = SearchLayer(view, scorer, entry, beam, layer, candidates);
                if (found > 0)
                {
                    entry = candidates[0];
                }
            }

            found = SearchLayerFiltered(view, scorer, entry, beam, candidates, isMatch, seedIds);

            using var resultHeap = DistanceHeap.Rent(Heaps.WorstKept, limit);
            for (int i = 0; i < found; i++)
            {
                int id = candidates[i];
                if (!isMatch(id))
                {
                    continue;
                }

                HnswNode? node = view.Nodes[id];
                if (node is null || node.Deleted)
                {
                    continue;
                }

                float score = quantized
                    ? ScoreQueryNode(effective, packedQuery, node)
                    : scorer.Score(node);
                resultHeap.PushBounded(limit, score, id);
            }

            int written = resultHeap.Count;
            int index = written;
            while (resultHeap.TryPop(out float score, out int id))
            {
                results[--index] = new SearchResult(id, score);
            }

            return written;
        }
        finally
        {
            if (normalized is not null)
            {
                ArrayPool<float>.Shared.Return(normalized);
            }

            if (packed is not null)
            {
                ArrayPool<ulong>.Shared.Return(packed);
            }

            if (queryCode is not null)
            {
                ArrayPool<byte>.Shared.Return(queryCode);
            }

            ArrayPool<int>.Shared.Return(candidates);
        }
    }

    /// <summary>
    /// Algorithm 5 restricted to layer 0 by a predicate. Non-matching nodes can still be visited
    /// (they are pushed to the exploration heap but never kept as results), which lets the beam
    /// walk through filtered-out regions to reach matching ones.
    /// </summary>
    private int SearchLayerFiltered(
        HnswGraphView view,
        QueryScorer scorer,
        int entryId,
        int ef,
        Span<int> destination,
        Func<int, bool> isMatch,
        IReadOnlyCollection<int>? seedIds)
    {
        (HeapOrder bestFirst, HeapOrder worstKept) = Heaps;
        using var candidates = DistanceHeap.Rent(bestFirst, ef);
        using var top = DistanceHeap.Rent(worstKept, ef);
        using var visited = VisitedSet.Rent(Math.Max(view.Count, 1));

        if (seedIds is not null)
        {
            int seeded = 0;
            foreach (int id in seedIds)
            {
                if (seeded >= ef)
                {
                    break;
                }

                if ((uint)id >= (uint)view.Count || !visited.Add(id))
                {
                    continue;
                }

                HnswNode? seed = view.Nodes[id];
                if (seed is null || seed.Deleted || !isMatch(id))
                {
                    continue;
                }

                float seedScore = scorer.Score(seed);
                candidates.Push(seedScore, id);
                top.PushBounded(ef, seedScore, id);
                seeded++;
            }
        }

        float worstScore = 0;
        bool hasWorst = top.TryPeekRoot(out worstScore, out _);
        if ((uint)entryId < (uint)view.Count && visited.Add(entryId))
        {
            HnswNode? entry = view.Nodes[entryId];
            if (entry is not null)
            {
                float entryScore = scorer.Score(entry);
                candidates.Push(entryScore, entryId);
                if (isMatch(entryId))
                {
                    top.PushBounded(ef, entryScore, entryId);
                    hasWorst = top.TryPeekRoot(out worstScore, out _);
                }
            }
        }

        while (candidates.Count > 0)
        {
            candidates.TryPeekRoot(out float candidateScore, out _);
            if (hasWorst && top.Count == ef && !Closer(candidateScore, worstScore))
            {
                break;
            }

            candidates.TryPop(out candidateScore, out int currentId);
            HnswNode? current = view.Nodes[currentId];
            if (current is null)
            {
                continue;
            }

            for (int source = 0; source < 2; source++)
            {
                int[] adjacent = source == 0 ? current.GetLinks(0) : current.GetBackbone(0);
                for (int i = 0; i < adjacent.Length; i++)
                {
                    int id = adjacent[i];
                    if ((uint)id >= (uint)view.Count || !visited.Add(id))
                    {
                        continue;
                    }

                    HnswNode? neighbor = view.Nodes[id];
                    if (neighbor is null || neighbor.Deleted)
                    {
                        continue;
                    }

                    float score = scorer.Score(neighbor);
                    if (isMatch(id))
                    {
                        if (!hasWorst || top.Count < ef || Closer(score, worstScore))
                        {
                            candidates.Push(score, id);
                            top.PushBounded(ef, score, id);
                            hasWorst = top.TryPeekRoot(out worstScore, out _);
                        }
                    }
                    else if (!hasWorst || Closer(score, worstScore))
                    {
                        // Filtered node: traverse it but never keep it as a hit.
                        candidates.Push(score, id);
                    }
                }
            }
        }

        int count = top.Count;
        int index = count;
        while (top.TryPop(out _, out int id))
        {
            destination[--index] = id;
        }

        return count;
    }

    private int EstimateResultCapacity(int k)
    {
        ValidateK(k);
        int live = LiveCount;
        return live == 0 ? 0 : Math.Min(k, live);
    }

    // -------------------------------------------------------------------------------------
    // Scoring / ordering helpers
    // -------------------------------------------------------------------------------------

    private (HeapOrder BestFirst, HeapOrder WorstKept) Heaps => _lowerIsCloser
        ? (HeapOrder.MinIdAscending, HeapOrder.MaxIdDescending)
        : (HeapOrder.MaxIdAscending, HeapOrder.MinIdDescending);

    private bool Closer(float candidate, float incumbent) =>
        _lowerIsCloser ? candidate < incumbent : candidate > incumbent;

    private int CompareByMetric((float Score, int Id) x, (float Score, int Id) y)
    {
        int byScore = _lowerIsCloser
            ? x.Item1.CompareTo(y.Item1)
            : y.Item1.CompareTo(x.Item1);
        return byScore != 0 ? byScore : x.Item2.CompareTo(y.Item2);
    }

    private bool Diverse(float distanceToQuery, float distanceToNeighbor) =>
        _lowerIsCloser
            ? distanceToQuery < _mc * distanceToNeighbor
            : distanceToQuery > distanceToNeighbor / _mc;

    private float ScoreId(int id, QueryScorer scorer, HnswGraphView view) =>
        scorer.Score(view.Nodes[id]!);

    private float ScoreQueryNode(ReadOnlySpan<float> query, ReadOnlySpan<ulong> packed, HnswNode node) => _distance switch
    {
        Distance.Cosine => _normalize
            ? 1f - VectorKernels.Dot(query, node.Vector.Floats)
            : VectorKernels.Cosine(query, node.Vector.Floats),
        Distance.DotProduct => VectorKernels.Dot(query, node.Vector.Floats),
        Distance.Euclidean => VectorKernels.EuclideanSquared(query, node.Vector.Floats),
        Distance.Hamming => BitKernels.Hamming(packed!, node.Bits!.Bits),
        _ => throw new InvalidOperationException($"Unknown distance metric '{_distance}'."),
    };

    private float ScoreNodeToNode(HnswNode a, HnswNode b) => _distance switch
    {
        Distance.Cosine => _normalize
            ? 1f - VectorKernels.Dot(a.Vector.Floats, b.Vector.Floats)
            : VectorKernels.Cosine(a.Vector.Floats, b.Vector.Floats),
        Distance.DotProduct => VectorKernels.Dot(a.Vector.Floats, b.Vector.Floats),
        Distance.Euclidean => VectorKernels.EuclideanSquared(a.Vector.Floats, b.Vector.Floats),
        Distance.Hamming => BitKernels.Hamming(a.Bits!.Bits, b.Bits!.Bits),
        _ => throw new InvalidOperationException($"Unknown distance metric '{_distance}'."),
    };

    private int AssignLevel(int id)
    {
        // 1 - U[0,1) maps to (0,1]; -ln(u) * 1/ln(M) is exponential with mean 1/(mL+1).
        double u = _seed.HasValue
            ? SeededUniform.Unit(_seed.Value, id)
            : 1.0 - Random.Shared.NextDouble();
        int level = (int)Math.Floor(-Math.Log(u) * _levelMultiplier);
        return Math.Clamp(level, 0, MaxSupportedLevel);
    }

    private void EnsureCapacity(int required)
    {
        if (required <= _nodes.Length)
        {
            return;
        }

        int newCapacity = (int)Math.Min(Math.Max((long)required, (long)_nodes.Length * 2), Array.MaxLength);
        if (_codes is not null && (long)newCapacity * _codeLength > Array.MaxLength)
        {
            throw new OutOfMemoryException($"HnswIndex cannot grow beyond {newCapacity} nodes for {_codeLength}-byte codes.");
        }

        var grown = new HnswNode?[newCapacity];
        Array.Copy(_nodes, grown, _count);
        _nodes = grown;

        if (_codes is not null)
        {
            var grownCodes = new byte[newCapacity * _codeLength];
            Array.Copy(_codes, grownCodes, (long)_count * _codeLength);
            _codes = grownCodes;
        }
    }

    private HnswGraphView CaptureView()
    {
        lock (_stateLock)
        {
            return new HnswGraphView(_nodes, _count, _entryPointId, _maxLevel);
        }
    }

    // -------------------------------------------------------------------------------------
    // Persistence support
    // -------------------------------------------------------------------------------------

    /// <summary>Whether the given id refers to a live (non-tombstoned) node.</summary>
    internal bool IsAlive(int id)
    {
        HnswGraphView view = CaptureView();
        return (uint)id < (uint)view.Count && view.Nodes[id] is { Deleted: false };
    }

    /// <summary>Whether any stored vector currently comes from a memory-mapped checkpoint page.</summary>
    internal bool UsesMappedVectors
    {
        get
        {
            HnswGraphView view = CaptureView();
            for (int id = 0; id < view.Count; id++)
            {
                HnswNode? node = view.Nodes[id];
                if (node is null)
                {
                    continue;
                }

                if (node.Vector.IsMapped || (node.Bits is { IsMapped: true }))
                {
                    return true;
                }
            }

            return false;
        }
    }

    /// <summary>Captures one node's stored vector and topology for a WAL record or checkpoint page.</summary>
    internal PersistedNode ExportNode(int id)
    {
        HnswGraphView view = CaptureView();
        if ((uint)id >= (uint)view.Count || view.Nodes[id] is not { } node)
        {
            throw new ArgumentOutOfRangeException(nameof(id), "unknown vector id");
        }

        byte[] vector = _distance == Distance.Hamming
            ? MemoryMarshal.AsBytes(node.Bits!.Bits).ToArray()
            : MemoryMarshal.AsBytes(node.Vector.Floats).ToArray();
        return new PersistedNode(
            id, node.Level, node.Deleted, vector,
            CloneTopology(node.Neighbors, node.Level),
            CloneTopology(node.Backbone, node.Level));
    }

    /// <summary>
    /// Replaces the whole index with a checkpoint/WAL snapshot. Vectors whose ids appear in
    /// <paramref name="vectorOffsets"/> are read directly from <paramref name="region"/>; all other
    /// nodes keep their own arrays.
    /// </summary>
    internal void Restore(
        IReadOnlyList<PersistedNode> nodes,
        MappedVectorRegion? region,
        IReadOnlyDictionary<int, long>? vectorOffsets)
    {
        lock (_deleteLock)
        {
            lock (_stateLock)
            {
                int maxId = -1;
                foreach (PersistedNode node in nodes)
                {
                    maxId = Math.Max(maxId, node.Id);
                }

                EnsureCapacity(maxId + 1);
                lock (_quantLock)
                {
                    _trainingSample.Clear();
                }

                var restored = new HnswNode?[_nodes.Length];
                int live = 0;
                int maxLevel = -1;
                int entry = -1;
                int count = maxId + 1;

                foreach (PersistedNode node in nodes)
                {
                    int level = Math.Clamp(node.Level, 0, MaxSupportedLevel);
                    VectorData vector;
                    VectorData? bits = null;
                    bool mapped = region is not null && vectorOffsets is not null &&
                                  vectorOffsets.TryGetValue(node.Id, out _);
                    long offset = mapped ? vectorOffsets![node.Id] : 0;

                    if (_distance == Distance.Hamming)
                    {
                        int words = _wordsPerVector;
                        if (node.Vector.Length != words * sizeof(ulong))
                        {
                            throw new InvalidDataException($"node {node.Id} has an invalid Hamming vector length");
                        }

                        bits = mapped && region is not null
                            ? VectorData.MappedBits(region, offset, words)
                            : VectorData.FromBits(MemoryMarshal.Cast<byte, ulong>(node.Vector).ToArray());
                        vector = VectorData.Empty;
                    }
                    else
                    {
                        if (node.Vector.Length != _dimensions * sizeof(float))
                        {
                            throw new InvalidDataException($"node {node.Id} has an invalid vector length");
                        }

                        vector = mapped && region is not null
                            ? VectorData.MappedFloats(region, offset, _dimensions)
                            : VectorData.FromFloats(MemoryMarshal.Cast<byte, float>(node.Vector).ToArray());
                    }

                    int[][] neighbors = NormalizeTopology(node.Neighbors, level, count);
                    int[][] backbone = NormalizeTopology(node.Backbone, level, count);
                    var restoredNode = new HnswNode(node.Id, level, vector, bits, level)
                    {
                        Deleted = node.Deleted,
                    };
                    restoredNode.SetTopology(neighbors, backbone);
                    restored[node.Id] = restoredNode;

                    if (!node.Deleted)
                    {
                        live++;
                    }

                    if (level > maxLevel)
                    {
                        maxLevel = level;
                        entry = node.Id;
                    }

                    if (_quantization is not null)
                    {
                        lock (_quantLock)
                        {
                            Quantizer? quantizer = Volatile.Read(ref _quantizer);
                            if (quantizer is null)
                            {
                                if (_trainingSample.Count < QuantizationRuntime.TrainingSampleSize)
                                {
                                    _trainingSample.Add(TrainingVector(node));
                                }
                            }
                            else if (_distance == Distance.Hamming)
                            {
                                quantizer.EncodePacked(bits!.Bits, _codes!.AsSpan(node.Id * _codeLength, _codeLength));
                            }
                            else
                            {
                                quantizer.Encode(vector.Floats, _codes!.AsSpan(node.Id * _codeLength, _codeLength));
                            }
                        }
                    }
                }

                _nodes = restored;
                _count = count;
                _entryPointId = entry;
                _maxLevel = maxLevel;
                Volatile.Write(ref _liveCount, live);
            }

            // Stored adjacency pages are validated and kept for the file-format contract, but a
            // WAL snapshot only records one node's links at insert time: reciprocal links added
            // to older nodes are not part of that record. Rebuilding from the stored vectors and
            // levels guarantees the restored graph is fully connected and free of tombstones.
            Compact();
        }
    }

    /// <summary>Updates a vector in place, preserving its id and graph topology.</summary>
    internal bool Replace(int id, ReadOnlySpan<float> vector)
    {
        ValidateVector(vector);
        HnswGraphView view = CaptureView();
        if ((uint)id >= (uint)view.Count || view.Nodes[id] is not { } node)
        {
            return false;
        }

        VectorData stored;
        VectorData? bits = null;
        if (_distance == Distance.Hamming)
        {
            var packed = new ulong[_wordsPerVector];
            BitVectors.PackSignBits(vector, packed);
            bits = VectorData.FromBits(packed);
            stored = VectorData.Empty;
        }
        else
        {
            var values = new float[_dimensions];
            if (_normalize)
            {
                Distances.CopyNormalized(vector, values);
            }
            else
            {
                vector.CopyTo(values);
            }

            stored = VectorData.FromFloats(values);
        }

        node.SetVector(stored, bits);
        if (_quantization is not null)
        {
            lock (_quantLock)
            {
                Quantizer? quantizer = Volatile.Read(ref _quantizer);
                if (quantizer is null)
                {
                    if (_trainingSample.Count < QuantizationRuntime.TrainingSampleSize)
                    {
                        _trainingSample.Add(_distance == Distance.Hamming ? vector.ToArray() : stored.Floats.ToArray());
                    }
                }
                else if (_distance == Distance.Hamming)
                {
                    quantizer.EncodePacked(bits!.Bits, _codes!.AsSpan(id * _codeLength, _codeLength));
                }
                else
                {
                    quantizer.Encode(stored.Floats, _codes!.AsSpan(id * _codeLength, _codeLength));
                }
            }
        }

        return true;
    }

    /// <summary>Rebinds all mapped vector slices after a checkpoint replaces the backing file.</summary>
    internal void RebindMappedVectors(MappedVectorRegion region)
    {
        HnswGraphView view = CaptureView();
        for (int id = 0; id < view.Count; id++)
        {
            view.Nodes[id]?.Rebind(region);
        }
    }

    private float[] TrainingVector(PersistedNode node)
    {
        if (_distance != Distance.Hamming)
        {
            return MemoryMarshal.Cast<byte, float>(node.Vector).ToArray();
        }

        var values = new float[_dimensions];
        BitVectors.UnpackSignBits(MemoryMarshal.Cast<byte, ulong>(node.Vector), _dimensions, values);
        return values;
    }

    private int[][] CloneTopology(int[][] source, int level)
    {
        var copy = new int[level + 1][];
        for (int i = 0; i <= level; i++)
        {
            copy[i] = i < source.Length ? (int[])source[i].Clone() : [];
        }

        return copy;
    }

    private static int[][] NormalizeTopology(int[][] source, int level, int count)
    {
        var result = new int[level + 1][];
        for (int i = 0; i <= level; i++)
        {
            int[] links = i < source.Length ? source[i] : [];
            var valid = new List<int>(links.Length);
            foreach (int id in links)
            {
                if ((uint)id < (uint)count && !valid.Contains(id))
                {
                    valid.Add(id);
                }
            }

            result[i] = valid.ToArray();
        }

        return result;
    }

    // -------------------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------------------

    private static void ValidateK(int k)
    {
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), k, "k must be positive.");
        }
    }

    private static void ValidateEf(int ef)
    {
        if (ef <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ef), ef, "efSearch must be positive.");
        }
    }

    private void ValidateQuery(ReadOnlySpan<float> query)
    {
        if (query.Length != _dimensions)
        {
            throw new ArgumentException($"query must have {_dimensions} dimensions, got {query.Length}.", nameof(query));
        }

        for (int i = 0; i < query.Length; i++)
        {
            if (!float.IsFinite(query[i]))
            {
                throw new ArgumentException($"query contains a non-finite value at index {i}.", nameof(query));
            }
        }
    }

    private void ValidateVector(ReadOnlySpan<float> vector)
    {
        if (vector.Length != _dimensions)
        {
            throw new ArgumentException($"vector must have {_dimensions} dimensions, got {vector.Length}.", nameof(vector));
        }

        for (int i = 0; i < vector.Length; i++)
        {
            if (!float.IsFinite(vector[i]))
            {
                throw new ArgumentException($"vector contains a non-finite value at index {i}.", nameof(vector));
            }
        }
    }

    /// <summary>
    /// Small stack-only wrapper that scores graph nodes against one query, either with full
    /// precision (construction and unquantized search) or via compact codes (quantized routing).
    /// </summary>
    private ref struct QueryScorer
    {
        private readonly HnswIndex _index;
        private readonly ReadOnlySpan<float> _query;
        private readonly ReadOnlySpan<ulong> _packed;
        private readonly ReadOnlySpan<byte> _code;
        private readonly ReadOnlySpan<byte> _codes;
        private readonly bool _quantized;

        internal QueryScorer(HnswIndex index, ReadOnlySpan<float> query, ReadOnlySpan<ulong> packed)
        {
            _index = index;
            _query = query;
            _packed = packed;
            _code = default;
            _codes = default;
            _quantized = false;
        }

        internal QueryScorer(HnswIndex index, ReadOnlySpan<byte> code, ReadOnlySpan<byte> codes)
        {
            _index = index;
            _query = default;
            _packed = default;
            _code = code;
            _codes = codes;
            _quantized = true;
        }

        internal float Score(HnswNode node) => _quantized
            ? _index.ScoreNodeQuantized(_code, _codes, node)
            : _index.ScoreQueryNode(_query, _packed, node);
    }
}
