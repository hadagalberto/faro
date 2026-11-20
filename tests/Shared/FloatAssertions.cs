using Xunit;

namespace Faro.Testing;

/// <summary>
/// Tolerance-based float equality for the test suites. xUnit's Assert.Equal compares floats
/// with a whole-number of decimal places; the suites need absolute tolerances instead.
/// </summary>
internal static class FloatAssertions
{
    /// <summary>Asserts that <paramref name="actual"/> is within <paramref name="tolerance"/> of <paramref name="expected"/>.</summary>
    public static void Equal(float expected, float actual, float tolerance)
    {
        Assert.True(
            MathF.Abs(expected - actual) <= tolerance,
            $"expected {expected} within tolerance {tolerance}, got {actual} (delta {actual - expected}).");
    }
}
