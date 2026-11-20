# Contributing to Faro

## Ground rules

- No stubs, TODO markers, `NotImplementedException`, skipped tests or placeholder assertions. Code that ships must be complete and tested.
- Tests assert real behaviour with known-valid and known-invalid vectors; randomized tests are compared against an independent model.
- If a test fails, fix the root cause. Do not weaken or delete a test to make it pass.
- Never publish: no `git push`, `gh`, `dotnet nuget push`, or package uploads. Commit locally with conventional commits.

## Setup

```bash
dotnet restore Faro.sln
dotnet build Faro.sln -c Release -warnaserror
dotnet test Faro.sln -c Release
```

Target frameworks are `net8.0` and `net10.0`; warnings are errors in `Directory.Build.props`. Both test projects use xUnit v3 with Microsoft.Testing.Platform. CI runs the same build and test commands on Ubuntu and Windows, plus a Native AOT publish of the RAG sample.

## Code style

- Follow `.editorconfig` (4 spaces, file-scoped namespaces, braces, XML docs on public APIs).
- Cite papers/algorithms in comments where an algorithm is non-obvious (HNSW, BM25, RRF, CRC32C).
- Keep `src/Faro` dependency-free. New dependencies for the core require an explicit decision.
- Prefer pooled buffers (`ArrayPool<T>`) and `ref struct` helpers in hot search paths; assert zero steady-state allocations when changing them.

## Adding a feature

1. Update `SPEC.md` only for scope decisions, not for routine work.
2. Add tests first where practical, including invalid inputs and tie/edge cases.
3. Run the full build and test suite on both target frameworks.
4. Update `README.md`, `README.pt-BR.md`, `docs/` and `CHANGELOG.md` to match the shipped behaviour.

## Commit format

Conventional commits: `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`, `chore: ...`, with a short imperative subject and an optional body explaining *why*.
