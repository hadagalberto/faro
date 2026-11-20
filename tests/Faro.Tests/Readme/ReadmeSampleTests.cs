using System.Text.Json.Serialization.Metadata;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace Faro.Tests.Readme;

/// <summary>
/// Compiles the README's database sample with Roslyn. The documented sample originally called
/// <c>GetOrCreateCollection("docs", ...)</c> (int keys) and then <c>Upsert("doc-1", ...)</c>
/// (string key), which does not compile. The test reads the real README file, so editing the sample
/// without keeping it valid fails CI.
/// </summary>
public class ReadmeSampleTests
{
    [Fact]
    public void DatabaseSample_Compiles()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryRoot(), "README.md"));
        int section = readme.IndexOf("## Database, persistence and durability", StringComparison.Ordinal);
        Assert.True(section >= 0);
        int fenceStart = readme.IndexOf("```csharp", section, StringComparison.Ordinal);
        int fenceEnd = readme.IndexOf("```", fenceStart + "```csharp".Length, StringComparison.Ordinal);
        Assert.True(fenceStart >= 0 && fenceEnd > fenceStart);

        string sample = readme[(fenceStart + "```csharp".Length)..fenceEnd]
            .Split('\n')
            // Drop namespace using directives only; keep `using var db = ...` declarations.
            .Where(line => !System.Text.RegularExpressions.Regex.IsMatch(line, @"^\s*using\s+[A-Za-z_][A-Za-z0-9_.]*\s*;\s*$"))
            .Aggregate(string.Empty, (all, line) => all + line + "\n");

        const string Prelude = """
            using System;
            using System.Collections.Generic;
            using System.Text.Json.Serialization.Metadata;
            using System.Threading.Tasks;
            using Faro;
            using Faro.Filtering;
            using Faro.Payload;

            public sealed class Doc { public string Tenant { get; set; } = ""; public string Text { get; set; } = ""; }
            internal sealed class DocContext
            {
                internal static Context Default => null!;
                internal sealed class Context { internal JsonTypeInfo<Doc> Doc => null!; }
            }
            internal static class Wrapper
            {
                internal static async Task Run()
                {
                    var schema = new PayloadSchema<Doc>().Keyword("tenant", d => d.Tenant).FullText("text", d => d.Text);
                    var embedding = new float[768];
                    var queryEmbedding = new float[768];
                    var doc = new Doc();
                    IEnumerable<(string, float[], Doc)> items = Array.Empty<(string, float[], Doc)>();
            """;

        string source = Prelude + sample + "\n}\n}\n";
        SyntaxTree tree = CSharpSyntaxTree.ParseText(source, cancellationToken: TestContext.Current.CancellationToken);
#pragma warning disable IL3000 // reference assemblies are loaded from disk in the test host, never single-file
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(assembly => !assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location))
            .Select(assembly => MetadataReference.CreateFromFile(assembly.Location))
            .Cast<MetadataReference>()
            .ToList();
#pragma warning restore IL3000

        CSharpCompilation compilation = CSharpCompilation.Create(
            "ReadmeSample", [tree], references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        var errors = compilation.GetDiagnostics(cancellationToken: TestContext.Current.CancellationToken)
            .Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)
            .Select(diagnostic => diagnostic.ToString())
            .ToArray();
        Assert.True(errors.Length == 0, string.Join(Environment.NewLine, errors));
    }

    [Fact]
    public void AllReadmeSamples_AvoidMismatchedKeyTypes()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryRoot(), "README.md"));
        int cursor = 0;
        int blocks = 0;
        while ((cursor = readme.IndexOf("```csharp", cursor, StringComparison.Ordinal)) >= 0)
        {
            int end = readme.IndexOf("```", cursor + "```csharp".Length, StringComparison.Ordinal);
            Assert.True(end > cursor);
            string block = readme[cursor..end];
            blocks++;
            // Hand-derived rule: the int-keyed overload has no string-keyed Upsert. A sample using
            // string keys must name the key type explicitly on GetOrCreateCollection.
            Assert.DoesNotContain("GetOrCreateCollection(\"", block);
            if (block.Contains("Upsert(\"", StringComparison.Ordinal))
            {
                Assert.Contains("GetOrCreateCollection<string,", block);
            }

            cursor = end + 3;
        }

        Assert.Equal(7, blocks); // every documented C# block was inspected
    }

    private static string RepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Faro.sln")))
        {
            dir = dir.Parent;
        }

        Assert.NotNull(dir);
        return dir.FullName;
    }
}
