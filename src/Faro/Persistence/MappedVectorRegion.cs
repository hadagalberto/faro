using System.IO.MemoryMappedFiles;

namespace Faro.Persistence;

/// <summary>
/// Read-only memory map over a checkpoint file. Vector pages are sliced directly from this view,
/// so searches read stored vectors from the OS page cache instead of copying them at open time.
/// </summary>
/// <remarks>
/// The mapping is owned by <see cref="FaroDb"/> and is replaced (after rebuilding affected indexes)
/// whenever a new checkpoint is installed. A caller must not use a span after the region is disposed.
/// </remarks>
internal sealed unsafe class MappedVectorRegion : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _accessor;
    private byte* _base;
    private bool _disposed;

    internal MappedVectorRegion(string path)
    {
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        try
        {
            if (stream.Length == 0)
            {
                throw new InvalidDataException("cannot map an empty checkpoint file");
            }

            Length = stream.Length;
            _file = MemoryMappedFile.CreateFromFile(
                stream, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
            _accessor = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            _base = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _base);
            if (_base is null)
            {
                throw new InvalidOperationException("failed to acquire memory-mapped view pointer");
            }
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    internal long Length { get; }

    internal bool IsDisposed => _disposed;

    internal ReadOnlySpan<float> Floats(long byteOffset, int count)
    {
        Validate(byteOffset, count, sizeof(float));
        return new ReadOnlySpan<float>(_base + byteOffset, count);
    }

    internal ReadOnlySpan<ulong> Ulongs(long byteOffset, int count)
    {
        Validate(byteOffset, count, sizeof(ulong));
        return new ReadOnlySpan<ulong>(_base + byteOffset, count);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_base is not null)
        {
            _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }

        _accessor.Dispose();
        _file.Dispose();
    }

    private void Validate(long byteOffset, int count, int elementSize)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (byteOffset < 0 || count < 0 || byteOffset > Length - (long)count * elementSize)
        {
            throw new InvalidDataException("vector page slice lies outside the memory-mapped file");
        }
    }
}
