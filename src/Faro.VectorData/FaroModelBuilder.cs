using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace Faro.VectorData;

/// <summary>
/// Builds the abstractions' provider-side record model, using the type validation rules of Faro's
/// payload schema and vector index.
/// </summary>
internal sealed class FaroModelBuilder : CollectionModelBuilder
{
    private const string SupportedDataTypes =
        "string, bool, byte, sbyte, short, ushort, int, uint, long, ulong, float, double, decimal and collections of string";

    private const string SupportedVectorTypes =
        "float[], IReadOnlyList<float>, IEnumerable<float>, ReadOnlyMemory<float> and Embedding<float>";

    public FaroModelBuilder()
        : base(new CollectionModelBuildingOptions
        {
            SupportsMultipleVectors = false,
            RequiresAtLeastOneVector = true,
            UsesExternalSerializer = true,
        })
    {
    }

    protected override bool IsDataPropertyTypeValid(Type type, out string supportedTypes)
    {
        supportedTypes = SupportedDataTypes;
        return IsNumeric(type) ||
               type == typeof(string) ||
               type == typeof(bool) ||
               (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>) &&
                IsDataPropertyTypeValid(type.GetGenericArguments()[0], out _)) ||
               IsStringCollection(type);
    }

    protected override bool IsVectorPropertyTypeValid(Type type, out string supportedTypes)
    {
        supportedTypes = SupportedVectorTypes;
        if (type == typeof(float[]) || type == typeof(ReadOnlyMemory<float>) || type == typeof(Memory<float>) ||
            type == typeof(IEnumerable<float>) || type == typeof(IReadOnlyList<float>) || type == typeof(IList<float>))
        {
            return true;
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Embedding<>))
        {
            return type.GetGenericArguments()[0] == typeof(float);
        }

        return type == typeof(Microsoft.Extensions.AI.Embedding);
    }

    protected override bool SupportsKeyAutoGeneration(Type keyPropertyType) => false;

    internal static bool IsNumeric(Type type)
    {
        Type t = Nullable.GetUnderlyingType(type) ?? type;
        return t == typeof(byte) || t == typeof(sbyte) || t == typeof(short) || t == typeof(ushort) ||
               t == typeof(int) || t == typeof(uint) || t == typeof(long) || t == typeof(ulong) ||
               t == typeof(float) || t == typeof(double) || t == typeof(decimal);
    }

    internal static bool IsStringCollection(Type type)
    {
        if (type == typeof(string))
        {
            return false;
        }

        if (type == typeof(string[]) || type == typeof(IEnumerable<string>) ||
            type == typeof(IReadOnlyList<string>) || type == typeof(IList<string>) ||
            type == typeof(List<string>))
        {
            return true;
        }

        return type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IEnumerable<>) &&
               type.GetGenericArguments()[0] == typeof(string);
    }
}
