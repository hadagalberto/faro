using Faro.Tests.Shared;

namespace Faro.Tests.Persistence;

/// <summary>Per-test scratch directory and fixture builders for persistence tests.</summary>
public abstract class PersistenceTestBase : IDisposable
{
    protected PersistenceTestBase()
    {
        Root = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "faro-m5", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Root);
    }

    protected string Root { get; }

    protected string DbPath(string name = "db.faro") => System.IO.Path.Combine(Root, name);

    protected static CollectionOptions<Doc> Options(int dimensions = 16)
    {
        var options = new CollectionOptions<Doc>
        {
            Dimensions = dimensions,
            Distance = Distance.DotProduct,
            Index = new HnswOptions
            {
                M = 8,
                EfConstruction = 64,
                EfSearch = 32,
                Seed = 0x5EED,
            },
            Serializer = TestFixtures.Serializer(),
            Schema = TestFixtures.Schema(),
            DefaultEfSearch = 32,
        };
        return options;
    }

    protected static float[] OneHot(int index, int dimensions = 16)
    {
        var vector = new float[dimensions];
        vector[index] = 1f;
        return vector;
    }

    protected static Doc MakeDoc(int i) => new()
    {
        Tenant = i % 2 == 0 ? "acme" : "globex",
        Year = 2020 + i,
        Active = i % 3 == 0,
        Tags = [i % 2 == 0 ? "red" : "blue"],
        Text = i % 4 == 0 ? "gato na arvore" : "cachorro e gato",
    };

    public void Dispose()
    {
        try
        {
            Directory.Delete(Root, recursive: true);
        }
        catch (IOException)
        {
            // Another process may still hold a file handle briefly; scratch directories are inert.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
