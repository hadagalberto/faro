using System.Globalization;
using Faro.Payload;
using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace Faro.VectorData;

/// <summary>Maps abstractions distance/index names and CLR property types onto Faro concepts.</summary>
internal static class FaroVectorDataMapping
{
    public static Distance ToDistance(string? distanceFunction, string collectionName)
    {
        return distanceFunction switch
        {
            null or "" or DistanceFunction.CosineSimilarity or DistanceFunction.CosineDistance => Distance.Cosine,
            DistanceFunction.DotProductSimilarity or DistanceFunction.NegativeDotProductSimilarity => Distance.DotProduct,
            DistanceFunction.EuclideanDistance or DistanceFunction.EuclideanSquaredDistance => Distance.Euclidean,
            DistanceFunction.HammingDistance => Distance.Hamming,
            _ => throw Unsupported(collectionName, $"distance function '{distanceFunction}'"),
        };
    }

    public static string DefaultDistanceFunction(Distance distance) => distance switch
    {
        Distance.Cosine => DistanceFunction.CosineSimilarity,
        Distance.DotProduct => DistanceFunction.DotProductSimilarity,
        Distance.Euclidean => DistanceFunction.EuclideanDistance,
        Distance.Hamming => DistanceFunction.HammingDistance,
        _ => throw new ArgumentOutOfRangeException(nameof(distance)),
    };

    /// <summary>Whether a score coming from Faro grows with relevance for the declared metric.</summary>
    public static bool HigherIsBetter(string? distanceFunction, Distance distance)
    {
        if (distanceFunction is DistanceFunction.CosineSimilarity or DistanceFunction.DotProductSimilarity)
        {
            return true;
        }

        return distance switch
        {
            Distance.Cosine => distanceFunction is null or "",
            Distance.DotProduct => false,
            _ => false,
        };
    }

    /// <summary>Converts a Faro distance value into the score exposed by the abstractions.</summary>
    public static double ToScore(float raw, string? distanceFunction, Distance distance)
    {
        return (distance, distanceFunction) switch
        {
            (Distance.Cosine, null or "" or DistanceFunction.CosineSimilarity) => 1.0 - raw,
            (Distance.Cosine, DistanceFunction.CosineDistance) => raw,
            (Distance.DotProduct, DistanceFunction.DotProductSimilarity) => raw,
            (Distance.DotProduct, DistanceFunction.NegativeDotProductSimilarity) => -raw,
            (Distance.DotProduct, null or "") => raw,
            _ => raw,
        };
    }

    public static void ValidateIndexKind(string? indexKind, string collectionName)
    {
        if (indexKind is null or "" or IndexKind.Hnsw or IndexKind.Flat or IndexKind.QuantizedFlat or IndexKind.Dynamic)
        {
            return;
        }

        throw Unsupported(collectionName, $"index kind '{indexKind}'");
    }

    public static FieldKind FieldKindFor(Type type)
    {
        Type t = Nullable.GetUnderlyingType(type) ?? type;
        if (t == typeof(string))
        {
            return FieldKind.Keyword;
        }

        if (t == typeof(bool))
        {
            return FieldKind.Bool;
        }

        if (FaroModelBuilder.IsNumeric(t))
        {
            return FieldKind.Numeric;
        }

        if (FaroModelBuilder.IsStringCollection(t))
        {
            return FieldKind.Tag;
        }

        throw new NotSupportedException($"payload field type '{type}' has no Faro field kind");
    }

    public static string FieldName(PropertyModel property) =>
        string.IsNullOrWhiteSpace(property.StorageName) ? property.ModelName : property.StorageName;

    public static double AsNumber(object? value) =>
        Convert.ToDouble(value ?? 0d, CultureInfo.InvariantCulture);

    public static VectorStoreException Unsupported(string collectionName, string detail) =>
        new($"Faro does not support {detail} (collection '{collectionName}')")
        {
            VectorStoreSystemName = "faro",
            CollectionName = collectionName,
        };
}
