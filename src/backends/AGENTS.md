# AGENTS.md — Backends

## Responsibility

Raw I/O operations for reading/writing tensor data. Backends know nothing about checkpoint formats.

## Key Types

- `SyncReader` / `SyncWriter` — blocking POSIX I/O
- `AsyncReader` / `AsyncWriter` — Tokio-based async
- `io_uring::Reader` / `io_uring::Writer` — Linux multi-worker rings
- `MmapReader` — memory-mapped file access
- Buffer pool in `mod.rs` (global, sharded)

## Conventions

- All public functions return `std::io::Result`
- Platform-specific code gated with `#[cfg(target_os = "linux")]`
- io_uring uses multi-worker architecture (not single shared ring)
- Buffer sizes are tuned for ML checkpoint workloads (≥ 1 MB)
- Batch operations coalesce adjacent reads

## Testing

```bash
cargo test --lib backends
```

## Do NOT

- Add format-specific logic here
- Change buffer pool parameters without benchmarking
- Use `unsafe` without documenting the safety invariant
