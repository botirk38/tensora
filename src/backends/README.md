# I/O Backends

High-performance I/O backends for tensor storage with platform-specific optimizations.

## Architecture

```
backends/
├── mod.rs           # Public API, platform-specific exports, buffer pool
├── io_uring.rs      # Linux io_uring backend (multi-worker submission)
├── async_io.rs      # Tokio async backend (cross-platform fallback)
├── sync_io.rs       # Synchronous std::fs backend
├── mmap.rs          # Memory-mapped I/O backend
├── odirect.rs       # O_DIRECT bypass backend (Linux)
├── batch.rs         # Batched read operations and coalescing
└── buffer_slice.rs  # Zero-copy buffer abstractions
```

## Backend Comparison

| Backend | Platform | Use Case | Key Features |
|---------|----------|----------|--------------|
| `sync` | All platforms | Smaller eager loads | Thread-parallel shard loading, dynamic chunking |
| `async` | All platforms | Range-heavy workloads | Tokio-based, grouped per-file tasks |
| `io_uring` | Linux 5.1+ | Larger eager loads | Multi-worker rings, dynamic planning |
| `mmap` | All platforms | Random access | OS-managed paging, lazy loading |

## Backend Selection

The `default` policy uses workload-aware heuristics:

### SafeTensors
- Single-shard or small multi-shard → `sync`
- Large multi-shard (≥ ~4 GB total) → `io_uring`

### ServerlessLLM
- Smaller range-heavy workloads → `async`
- Larger partitioned workloads → `io_uring`

## Backend Details

### io_uring Backend

The Linux backend uses multiple worker-owned rings rather than a single shared ring. This architectural change was key to making it competitive on H100 hardware.

**Features**:
- Multi-worker ring submission
- Dynamic ring sizing based on workload
- Batched submission with completion coalescing
- Direct-I/O support for aligned reads

**Requirements**:
- Linux kernel 5.1+
- `io-uring` crate

### Tokio Backend

Cross-platform async I/O using Tokio's filesystem operations.

**Features**:
- Works on Linux, macOS, Windows
- Integrates with existing Tokio applications
- Grouped per-file task execution

**Platform behavior**:
- On non-Linux platforms, async delegates to sync via `spawn_blocking`, matching sync performance while preserving the async interface.

### Sync Backend

Blocking I/O using standard library filesystem operations with dynamic chunking and thread-parallel shard loading.

**When to use**:
- Smaller eager checkpoint loads
- CLI tools without async
- When simplicity is preferred

### Memory-Mapped Backend

Uses OS virtual memory to map files directly into address space.

**When to use**:
- Read-only access patterns
- Random access to file regions
- When only specific tensors are needed

## Buffer Pool

The module includes a global buffer pool optimized for ML checkpoint loading:

```rust
use tensora::backends::get_buffer_pool;

let pool = get_buffer_pool();
let buffer = pool.get(size);
```

**Pool configuration**:
- 8 shards for reduced contention
- 4 buffers per thread-local cache
- 32 max buffers per shard
- 1MB minimum buffer size

## Error Handling

All backends return `std::io::Result`:

```rust
match backends::load("model.safetensors").await {
    Ok(data) => process(data),
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
        eprintln!("File not found");
    }
    Err(e) => return Err(e.into()),
}
```

## Testing

```bash
# Run backend tests
cargo test --lib backends

# Run with specific backend tests
cargo test io_uring  # Linux only
cargo test async_io
cargo test sync_io
cargo test mmap
```
