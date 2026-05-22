# I/O Backends

High-performance I/O backends for tensor checkpoint loading with platform-specific optimizations.

## Modules

| File | Description | Platform |
|------|-------------|----------|
| `sync_io.rs` | Thread-parallel POSIX reads with dynamic chunking | All |
| `async_io.rs` | Tokio-based async I/O | All |
| `io_uring.rs` | Multi-worker io_uring submission | Linux 5.1+ |
| `mmap.rs` | Memory-mapped lazy loading | All |
| `odirect.rs` | O_DIRECT bypass for aligned reads | Linux |
| `batch.rs` | Batched read operations and coalescing | All |
| `buffer_slice.rs` | Zero-copy buffer abstractions | All |
| `byte.rs` | Byte-level utilities | All |
| `availability.rs` | Runtime backend detection | All |
| `mod.rs` | Public API, buffer pool, platform exports | All |

## Backend Selection Heuristics

The adaptive `default` selects automatically:

- **SafeTensors:** single/small-shard → `sync`; large multi-shard (≥ 4 GB) → `io_uring`
- **ServerlessLLM:** range-heavy → `async`; large partitioned → `io_uring`
- **Non-Linux:** falls back to `sync` or `async`

## Buffer Pool

Global pool with 8 shards, 4 thread-local buffers, 32 max per shard, 1 MB minimum size.

## Testing

```bash
cargo test --lib backends
```
