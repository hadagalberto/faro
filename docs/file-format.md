# Faro file format (design)

> **Not implemented in Milestone 1.** Faro currently has no persistence layer. This document is the contract the persistence milestone must implement; it is kept in the repository so the on-disk layout is reviewed before any bytes are written.

## Goals

- One `.faro` file per database, memory-mapped for vector reads.
- Versioned and self-describing enough to reject unsupported versions cleanly.
- Crash-safe: committed operations survive; torn trailing writes are ignored on open.
- Deterministic page offsets so two databases with the same operations produce the same bytes.

## Top-level layout

| Region | Contents |
| --- | --- |
| Header | Magic `FARO`, format version, header checksum, page-table offset/count, database id |
| Page table | Fixed-size records: page kind, offset, length, CRC32C of the page |
| Vector pages | Paged `float32` blocks, one run per collection, 64-byte aligned |
| Graph pages | HNSW adjacency lists and level metadata |
| Payload pages | Source-generated JSON payloads (AOT-safe) with a length prefix |
| Text pages | BM25 postings and document lengths |
| WAL | Append-only records for upserts/deletes with per-record CRC32C |

All integers are little-endian. Padding bytes are zero. The header is written last during checkpoint creation, so an incomplete checkpoint is never mistaken for a committed one.

## Checksums

CRC32C (Castagnoli polynomial, reflected form) is used for the header and every page/record. A page whose stored checksum does not match its contents is treated as absent. A torn WAL tail — a partial record or a checksum mismatch at the end — is truncated on open; records before it are replayed in order.

## Durability modes

| Mode | Behaviour |
| --- | --- |
| `Always` | fsync after every WAL record; slowest, strongest guarantee |
| `Batched` | fsync once per batch/commit group (default) |
| `None` | rely on OS flushing; fastest, loses recent commits on power loss |

## Checkpointing

Checkpoints write a new file to a temporary name, fsync it, then atomically rename over the main file (on Windows, replace via `File.Move(..., overwrite: true)` after closing all mappings). The WAL is truncated only after the rename is durable. Opening a database always: read header → validate pages → replay WAL → truncate torn tail.

## In-memory mode

`FaroDb.InMemory()` uses the same in-memory structures with no file and no WAL. It is the mode used by tests that do not exercise durability.
