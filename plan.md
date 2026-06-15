# Tensora Storage Refactor Plan

The storage refactor replaces legacy backend vocabulary and request-wrapper APIs with direct storage-engine traits and explicit domain types. This is intentionally a breaking API cleanup; no compatibility layer is maintained.

## Public API

```rust
pub trait StorageEngine {
    const KIND: StorageKind;

    fn kind(&self) -> StorageKind;
    fn availability() -> StorageAvailability
    where
        Self: Sized;
}

pub trait ReadableStorage: StorageEngine {
    fn read_file(&self, path: &Path) -> IoResult<OwnedBytes>;
    fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes>;
    fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>>;
}

pub trait WritableStorage: StorageEngine {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()>;
    fn write_slices(&mut self, writes: &[WriteSlice<'_>]) -> IoResult<()>;
    fn set_len(&mut self, len: u64) -> IoResult<()>;
    fn sync_data(&mut self) -> IoResult<()>;
    fn sync_all(&mut self) -> IoResult<()>;
    fn finish(self) -> IoResult<()>;
}

pub trait MappableStorage: StorageEngine {
    fn map_file(&self, path: &Path) -> IoResult<MmapRegion>;
    fn map_range(&self, path: &Path, range: ByteRange) -> IoResult<MmapRegion>;
}
```

Tokio exposes inherent async methods and implements explicit async traits:

```rust
pub trait AsyncReadableStorage: StorageEngine {
    async fn read_file(&self, path: &Path) -> IoResult<OwnedBytes>;
    async fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes>;
    async fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>>;
}

pub trait AsyncWritableStorage: StorageEngine {
    async fn write_all_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()>;
    async fn write_slices(&mut self, writes: &[WriteSlice<'_>]) -> IoResult<()>;
    async fn set_len(&mut self, len: u64) -> IoResult<()>;
    async fn sync_data(&mut self) -> IoResult<()>;
    async fn sync_all(&mut self) -> IoResult<()>;
    async fn finish(self) -> IoResult<()>;
}
```

## Vocabulary Types

- `ByteRange` — validated half-open range `[start, end)`.
- `FileRange` — path plus `ByteRange` for batch reads.
- `RangeRead` — result from `read_ranges`, preserving `request_index` and `range`.
- `WriteMode` — `CreateNew`, `CreateOrTruncate`, `OpenExisting`.
- `WriteOptions` — creation mode, parent-directory creation, optional preallocation.
- `WriteSlice` — offset plus byte slice for ordered multi-write operations.

## Engine Shape

| Engine           | StorageEngine | ReadableStorage | WritableStorage | MappableStorage |
|------------------|:-------------:|:---------------:|:---------------:|:---------------:|
| `SyncStorage`    | ✓ | ✓ | via `SyncWriter` | — |
| `TokioStorage`   | ✓ | async inherent + async trait | via `TokioWriter` | — |
| `MmapStorage`    | ✓ | — | — | ✓ |
| `IoUringStorage` | ✓ | ✓ | via `IoUringWriter` | — |

Concrete writer creation stays on writer types:

```rust
SyncWriter::create(path, WriteOptions::create_or_truncate())?;
TokioWriter::create(path, WriteOptions::create_or_truncate()).await?;
IoUringWriter::create(path, WriteOptions::create_or_truncate())?;
```

## Implementation Rules

- No reader-opening method on the public traits.
- No generic private file-wrapper abstraction for direct/buffered read state.
- Keep direct file-open/read logic inside trait methods when straightforward.
- Keep helpers only where they encode real complexity:
  - Linux O_DIRECT alignment/exact-read logic.
  - Linux chunked parallel full-file reads.
  - io_uring submit/wait and exact positioned I/O loops.
- Tokio delegates blocking reads to `SyncStorage` instead of duplicating Linux O_DIRECT logic.
- Storage engines handle raw I/O only and remain format-unaware.

## Verification

```bash
cargo fmt --all
cargo check --lib --locked
cargo check --benches --locked
cargo test --lib --locked
cargo test --bins --locked
cargo clippy --lib --locked -- -D warnings
cargo build --locked
```
