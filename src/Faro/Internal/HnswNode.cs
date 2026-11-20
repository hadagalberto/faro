namespace Faro.Internal;

/// <summary>
/// A single node in the HNSW graph. Each node owns its stored vector, its assigned layer, an
/// immutable-snapshot per-layer adjacency list and a fine-grained gate used to serialise
/// structural updates to that node only.
/// </summary>
/// <remarks>
/// <para>
/// Both adjacency arrays are copy-on-write: they are only ever replaced with a new array under
/// <see cref="Gate"/>, never mutated in place. A reader that grabs a snapshot once therefore
/// observes a fully consistent view of every layer without taking a lock, which is what lets
/// searches run concurrently with inserts and deletes.
/// </para>
/// <para>
/// <see cref="Neighbors"/> holds the Algorithm 4 links (bounded by M / M0). <see cref="Backbone"/>
/// is a separate connectivity spine: the nearest neighbour chosen when a node joins a layer is
/// linked bidirectionally, and those edges are exempt from heuristic pruning. This guarantees
/// the live graph stays reachable from the entry point even where the diversity heuristic would
/// otherwise strip a node's last reverse link.
/// </para>
/// <para>
/// <see cref="Deleted"/> is a tombstone flag: a deleted node keeps its own adjacency lists so it
/// can still act as a traversal entry point, but searches never return it and never expand
/// through a deleted neighbour. Links <em>to</em> deleted nodes are removed by neighbour repair
/// and vanish entirely after <c>Compact()</c>.
/// </para>
/// </remarks>
internal sealed class HnswNode
{
    internal readonly int Id;
    internal readonly int Level;
    internal readonly float[] Vector;
    internal readonly ulong[]? Bits;
    internal readonly object Gate = new();

    private int[][] _neighbors;
    private int[][] _backbone;
    private int _deleted;

    internal HnswNode(int id, int level, float[] vector, ulong[]? bits, int layers)
    {
        Id = id;
        Level = level;
        Vector = vector;
        Bits = bits;
        _neighbors = CreateEmpty(layers);
        _backbone = CreateEmpty(layers);
    }

    private static int[][] CreateEmpty(int layers)
    {
        var store = new int[layers + 1][];
        for (int i = 0; i <= layers; i++)
        {
            store[i] = [];
        }

        return store;
    }

    internal bool Deleted
    {
        get => Volatile.Read(ref _deleted) != 0;
        set => Volatile.Write(ref _deleted, value ? 1 : 0);
    }

    /// <summary>Returns a consistent snapshot of all per-layer Algorithm 4 adjacency arrays.</summary>
    internal int[][] Neighbors => Volatile.Read(ref _neighbors);

    /// <summary>Returns a consistent snapshot of all per-layer backbone adjacency arrays.</summary>
    internal int[][] Backbone => Volatile.Read(ref _backbone);

    /// <summary>Returns the main adjacency snapshot for one layer (never mutated in place).</summary>
    internal int[] GetLinks(int layer) => Neighbors[layer];

    /// <summary>Returns the backbone adjacency snapshot for one layer (never mutated in place).</summary>
    internal int[] GetBackbone(int layer) => Backbone[layer];

    internal void SetLinks(int layer, int[] ids)
    {
        lock (Gate)
        {
            Replace(ref _neighbors, layer, ids);
        }
    }

    internal void AddLink(int layer, int id)
    {
        lock (Gate)
        {
            if (Array.IndexOf(_neighbors[layer], id) < 0)
            {
                Replace(ref _neighbors, layer, Append(_neighbors[layer], id));
            }
        }
    }

    internal void RemoveLinks(int layer, int id) => Remove(ref _neighbors, layer, id);

    internal void AddBackbone(int layer, int id)
    {
        lock (Gate)
        {
            if (Array.IndexOf(_backbone[layer], id) < 0)
            {
                Replace(ref _backbone, layer, Append(_backbone[layer], id));
            }
        }
    }

    internal void RemoveBackbone(int layer, int id) => Remove(ref _backbone, layer, id);

    private void Remove(ref int[][] store, int layer, int id)
    {
        lock (Gate)
        {
            int[] current = store[layer];
            int found = 0;
            for (int i = 0; i < current.Length; i++)
            {
                if (current[i] == id)
                {
                    found++;
                }
            }

            if (found == 0)
            {
                return;
            }

            var updated = new int[current.Length - found];
            int next = 0;
            for (int i = 0; i < current.Length; i++)
            {
                if (current[i] != id)
                {
                    updated[next++] = current[i];
                }
            }

            Replace(ref store, layer, updated);
        }
    }

    // Called with Gate held. ref to the field keeps both stores on one implementation.
    private static void Replace(ref int[][] store, int layer, int[] links)
    {
        var snapshot = (int[][])store.Clone();
        snapshot[layer] = links;
        Volatile.Write(ref store, snapshot);
    }

    private static int[] Append(int[] current, int id)
    {
        var updated = new int[current.Length + 1];
        Array.Copy(current, updated, current.Length);
        updated[current.Length] = id;
        return updated;
    }
}

/// <summary>
/// An immutable snapshot of the graph topology captured at the start of an operation. Holding
/// the array reference and the entry-point metadata together keeps searches consistent with
/// inserts that grow the backing array.
/// </summary>
internal readonly struct HnswGraphView
{
    internal readonly HnswNode?[] Nodes;
    internal readonly int Count;
    internal readonly int EntryPointId;
    internal readonly int MaxLevel;

    internal HnswGraphView(HnswNode?[] nodes, int count, int entryPointId, int maxLevel)
    {
        Nodes = nodes;
        Count = count;
        EntryPointId = entryPointId;
        MaxLevel = maxLevel;
    }
}
