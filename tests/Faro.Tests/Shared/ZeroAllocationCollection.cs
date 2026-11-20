using Xunit;

namespace Faro.Tests;

/// <summary>
/// Allocation measurements run alone, never alongside other test collections. Even though the
/// runner configuration already disables collection parallelism, the explicit collection keeps
/// the guarantee local to the tests that depend on it if that configuration is ever changed.
/// </summary>
[CollectionDefinition("ZeroAllocation", DisableParallelization = true)]
public sealed class ZeroAllocationCollection
{
}
