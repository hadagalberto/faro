using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace Faro.VectorData;

/// <summary>Converts the native vector shapes of the abstractions into Faro's float spans.</summary>
internal static class FaroVectorConverter
{
    public static async ValueTask<float[]> ToVectorAsync<TInput>(
        TInput value,
        VectorPropertyModel property,
        string collectionName,
        CancellationToken cancellationToken)
    {
        if (value is null)
        {
            throw FaroVectorDataMapping.Unsupported(collectionName, "a null vector value");
        }

        if (TryGetNativeVector(value, out float[]? native))
        {
            return native!;
        }

        if (property.EmbeddingGenerationDispatcher is not null)
        {
            Microsoft.Extensions.AI.Embedding embedding =
                await property.GenerateEmbeddingAsync(value, cancellationToken).ConfigureAwait(false);
            if (TryGetNativeVector(embedding, out float[]? generated))
            {
                return generated!;
            }
        }

        throw FaroVectorDataMapping.Unsupported(
            collectionName, $"vector value of type '{typeof(TInput)}' (expected float[], ReadOnlyMemory<float> or Embedding<float>)");
    }

    private static bool TryGetNativeVector(object value, out float[]? vector)
    {
        switch (value)
        {
            case float[] array:
                vector = array;
                return true;
            case ReadOnlyMemory<float> readOnly:
                vector = readOnly.ToArray();
                return true;
            case Memory<float> memory:
                vector = memory.ToArray();
                return true;
            case IReadOnlyList<float> list:
                vector = ToArray(list);
                return true;
            case IEnumerable<float> enumerable:
                vector = [.. enumerable];
                return true;
            case Embedding<float> embedding:
                vector = embedding.Vector.ToArray();
                return true;
            default:
                vector = null;
                return false;
        }
    }

    private static float[] ToArray(IReadOnlyList<float> list)
    {
        var result = new float[list.Count];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = list[i];
        }

        return result;
    }
}
