namespace Faro;

/// <summary>The kind of vector quantization applied to a collection.</summary>
public enum QuantizationKind
{
    /// <summary>Symmetric per-dimension signed int8 quantization (one byte per dimension).</summary>
    Int8,

    /// <summary>Binary sign quantization (one bit per dimension, packed into 64-bit words).</summary>
    Binary,
}
