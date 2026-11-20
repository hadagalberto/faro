# Faro file format (version 1)

Faro persists one database as two files: a checkpoint (`knowledge.faro`) and its write-ahead log (`knowledge.faro.wal`). Both are versioned, little-endian and checksummed with CRC32C (Castagnoli, reflected polynomial `0x82F63B78`). The layout below is what the code in `src/Faro/Persistence` reads and writes; changing a byte offset requires a format version bump.

## Goals

- One `.faro` file per database, with vector pages read through a memory map.
- Self-describing enough to reject unsupported versions and damaged pages cleanly.
- Crash-safe: committed WAL records survive; a torn trailing write is ignored and truncated on open.
- Deterministic: the same operation log on the same options produces byte-identical checkpoint files.

## Top-level layout

| Region | Contents |
| --- | --- |
| Header | Magic `FARO`, version, header checksum, database id, page-table offset/count |
| Page table | Fixed-size records: page kind, collection id, offset, length, CRC32C |
| Vector pages | Stored `float32` vectors (or packed sign bits for Hamming), 64-byte aligned |
| Graph pages | HNSW node levels and per-layer adjacency lists |
| Payload pages | Length-prefixed payload bytes and their encoded user keys |
| Text pages | BM25 postings and document lengths (validated for integrity) |
| WAL | Separate `.wal` file: append-only typed records with a CRC32C per record |

All integers are little-endian. Padding bytes are zero. Checkpoints are built in a temporary file: the header is written first, then the page table and pages, and the file is flushed before an atomic rename publishes it. Readers therefore only ever see a fully written checkpoint at the final path.

## Checkpoint header (128 bytes)

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | Magic `FARO` |
| 4 | 4 | Format version (currently `1`) |
| 8 | 4 | Header size (128) |
| 12 | 4 | Reserved (zero) |
| 16 | 8 | Database id: FNV-1a of all serialized collection definitions |
| 24 | 8 | Reserved (zero) |
| 32 | 8 | Page-table offset (128) |
| 40 | 4 | Page count |
| 44 | 4 | Collection count |
| 48 | 16 | Reserved (zero) |
| 64 | 4 | CRC32C over bytes `[0,64)` |
| 68 | 60 | Reserved (zero) |

A missing file, a bad magic/version/header size or a header checksum mismatch raises `InvalidDataException` on open. A valid file with a bad page checksum simply treats that page as absent.

## Page table

`pageCount` fixed 32-byte records immediately after the header:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 1 | Page kind |
| 1 | 3 | Reserved (zero) |
| 4 | 4 | Collection id |
| 8 | 8 | Absolute file offset |
| 16 | 8 | Page length |
| 24 | 4 | CRC32C of the page contents |
| 28 | 4 | Reserved (zero) |

Page kinds: `1` collection definition, `2` vectors, `3` graph, `4` payloads, `5` text. Pages are written in collection-id order and each starts at a 64-byte-aligned offset. A collection is loaded only when its definition, vector and graph pages are all present and valid.

### Page contents

- **Collection definition**: collection id, name, key kind (`int32=1`, `int64=2`, `guid=3`, `string=4`), dimensions, distance, normalization flag, HNSW tunables (`M`, `EfConstruction`, `EfSearch`, optional seed/level multiplier, Algorithm 4 flags, `Mc`), quantization kind/rescore/quantile, BM25 `k1`/`b`, declared field names and kinds, and the full-text field name.
- **Vector page**: node count, then per node: id (`int32`), level (`int32`), deleted flag (`byte`), length-prefixed stored vector. Non-Hamming vectors are `dimensions × 4` bytes of `float32`; Hamming vectors are `ceil(dimensions/64) × 8` bytes of packed sign bits.
- **Graph page**: node count, then per node: id and its neighbour/backbone layers (layer count, then each layer's id count and ids).
- **Payload page**: record count, then per record: internal vector id, length-prefixed encoded key, length-prefixed serialized payload (UTF-8 JSON produced by a source-generated `JsonTypeInfo<T>`).
- **Text page**: document count, then per document: id, token count, `(term, tf)` pairs, and the document length. BM25 is rebuilt from the typed payloads on materialisation so replay never needs a user-supplied serializer or tokenizer.

HNSW adjacency pages are validated and stored for tooling, but the in-memory graph is rebuilt from the stored vectors and levels on open. Reciprocal links created by later inserts are not part of an earlier record's snapshot; rebuilding guarantees a connected graph and removes dangling tombstones.

## Write-ahead log

The WAL begins with a 16-byte header: magic `FAROWAL1`, version `1`, and CRC32C of the first 12 bytes. Each record is framed as:

| Field | Size | Meaning |
| --- | --- | --- |
| length | 4 | Payload byte count (excludes the type byte) |
| crc32c | 4 | CRC32C over `(type byte || payload)` |
| type | 1 | Record kind |
| payload | length | Typed payload |

Record types:

| Type | Payload |
| --- | --- |
| `1` | Collection id + serialized collection definition |
| `2` | Collection id + full node snapshot (vector, level, topology) + full stored record (key and payload bytes) |
| `3` | Collection id + internal id of a deleted node |
| `4` | Collection id (delete collection) |

Records are self-contained: an upsert carries the complete vector, payload and key, so replaying any prefix of well-formed records yields a consistent prefix of committed operations. A partial frame, a length that runs past the end of the file, or a CRC mismatch marks a torn tail. The scan stops there; every earlier record is replayed in order, and the file is truncated to the last valid frame before new writes are appended.

## Durability modes

| Mode | Behaviour |
| --- | --- |
| `Always` | `fsync` after every WAL record; slowest, strongest guarantee |
| `Batched` | `fsync` once per commit group (explicit `Flush`, dispose, checkpoint or batch) — default |
| `None` | rely on OS flushing; fastest, loses recent commits on power loss |

`Flush()` and `FlushAsync()` force a durable flush in `Batched` mode; they are no-ops in `None` mode and redundant in `Always` mode.

## Checkpointing

`FaroDb.Checkpoint()` collects the current state of every materialised collection, writes all pages to `<path>.tmp`, flushes it to disk, then atomically replaces the main file (`File.Move(..., overwrite: true)`). The WAL is reset only after the rename is durable. The old memory map is released before the replace and a new one is created afterwards; mapped vector slices are rebound to it. Because spans can point into the old mapping, callers must not search concurrently with a checkpoint on the same database instance.

## In-memory mode

`FaroDb.InMemory()` uses the same in-memory structures with no file, no WAL and no memory map. It is the mode used by tests that do not exercise durability; `Checkpoint()` throws in this mode.

## References

- Castagnoli et al., "Optimization of Cyclic Redundancy-Check Codes with 24 and 32 Parity Bits" (1993) — CRC32C.
- Malkov & Yashunin (2016/2018) — the HNSW topology stored in graph pages.
