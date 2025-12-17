# Backends Module

High-performance I/O backends for tensor storage with platform-specific optimizations.

## Overview

The backends module provides zero-copy I/O operations optimized for loading large tensor files. The API is async-first with explicit sync alternatives for blocking contexts.

## Architecture

```
backends/
├── mod.rs           # Public API, platform-specific exports, buffer pool
├── io_uring.rs      # Linux io_uring backend (primary, high-performance)
├── async_io.rs      # Tokio async backend (cross-platform fallback)
├── sync_io.rs       # Synchronous std::fs backend
├── mmap.rs          # Memory-mapped I/O backend
├── odirect.rs       # O_DIRECT bypass backend (Linux)
├── batch.rs         # Batched read operations
└── buffer_slice.rs  # Zero-copy buffer abstractions
```

## Backend Comparison

| Backend | Platform | Use Case | Key Features |
|---------|----------|----------|--------------|
| `io_uring` | Linux 5.1+ | Production loading | Zero-copy, batched ops, kernel-offloaded |
| `async_io` | All platforms | Cross-platform apps | Tokio compatibility, async runtime |
| `sync_io` | All platforms | Simple scripts | No async runtime needed |
| `mmap` | All platforms | Read-only access | OS-managed paging, lazy loading |
| `odirect` | Linux | Large files | Bypasses page cache |

## Usage

### Async Operations (Recommended)

```rust
use tensor_store::backends;

// Load entire file (auto-selects best backend)
let data = backends::load("model.safetensors").await?;

// Parallel loading with N chunks
let data = backends::load_parallel("model.safetensors", 4).await?;

// Load specific byte range
let chunk = backends::load_range("model.safetensors", offset, length).await?;

// Batch load multiple ranges
let ranges = vec![
    ReadRequest::new("file1.bin", 0, 1024),
    ReadRequest::new("file2.bin", 512, 2048),
];
let results = backends::load_batch(ranges).await?;

// Write data
backends::write_all("output.bin", data).await?;
```

### Synchronous Operations

```rust
use tensor_store::backends::sync;

// For blocking contexts (no async runtime)
let data = sync::load("model.safetensors")?;
let chunk = sync::load_range("model.safetensors", 1024, 512)?;
sync::write_all("output.bin", data)?;
```

### Memory-Mapped I/O

```rust
use tensor_store::backends::mmap;

// Memory-map for read-only access
let mmap = mmap::load("model.safetensors")?;
let slice = &mmap[offset..offset + length];
```

### O_DIRECT Operations (Linux)

```rust
use tensor_store::backends::odirect;

// Bypass page cache for large files
let data = odirect::load_aligned("large_model.bin")?;
```

## Backend Details

### io_uring Backend (`io_uring.rs`)

The primary backend for Linux systems, using the io_uring kernel interface for maximum performance.

**Features**:
- Zero-copy I/O with pre-registered buffers
- Batched submission of multiple I/O operations
- Kernel-offloaded completion notification
- Single file handle reuse for parallel reads

**Requirements**:
- Linux kernel 5.1+
- `tokio-uring` runtime

**Performance characteristics**:
- Best for large files (>10MB)
- Scales well with parallel chunks
- ~5% faster than Tokio for sequential reads
- Up to 70% faster with buffer pooling

### Tokio Backend (`async_io.rs`)

Cross-platform async I/O using Tokio's filesystem operations.

**Features**:
- Works on Linux, macOS, Windows
- Integrates with existing Tokio applications
- Parallel loading support

**When to use**:
- Non-Linux platforms
- Applications already using Tokio runtime
- When cross-platform compatibility is needed

### Sync Backend (`sync_io.rs`)

Blocking I/O using standard library filesystem operations.

**Features**:
- No async runtime required
- Simple, predictable behavior
- Good for small files and scripts

**When to use**:
- CLI tools without async
- Small file operations
- Testing and prototyping

### Memory-Mapped Backend (`mmap.rs`)

Uses OS virtual memory to map files directly into address space.

**Features**:
- Lazy loading (pages loaded on access)
- OS-managed memory caching
- Zero-copy access to file data

**When to use**:
- Read-only access patterns
- Random access to file regions
- When file fits in address space

**Considerations**:
- Page faults add latency on first access
- Not suitable for frequent writes
- Memory pressure can cause eviction

### O_DIRECT Backend (`odirect.rs`)

Direct I/O bypassing the kernel page cache.

**Features**:
- Avoids double-buffering in page cache
- Predictable memory usage
- Better for large, one-time reads

**Requirements**:
- Linux only
- 512-byte aligned buffers and offsets
- `OwnedAlignedBuffer` for safe alignment

**When to use**:
- Loading large models (>RAM size)
- When page cache pollution is a concern
- Single-use data that won't be re-read

## Buffer Pool

The module includes a global buffer pool optimized for ML checkpoint loading:

```rust
use tensor_store::backends::get_buffer_pool;

let pool = get_buffer_pool();
let buffer = pool.alloc(1024 * 1024); // 1MB buffer
```

**Pool configuration**:
- 8 shards for reduced contention
- 4 buffers per thread-local cache
- 32 max buffers per shard
- 1MB minimum buffer size
- Pinned memory to avoid page faults

**Performance impact**:
- 70% speedup over non-pooled allocation
- Reduces allocation overhead in hot paths
- Reuses buffers across tensor loads

## Zero-Copy Architecture

### BufferSlice (`buffer_slice.rs`)

Safe abstraction for non-overlapping buffer slices:

```rust
use tensor_store::backends::buffer_slice::BufferSlice;

// Pre-allocate final buffer
let mut buffer = vec![0u8; total_size];

// Split into non-overlapping slices for parallel tasks
let slices = BufferSlice::split(&mut buffer, chunk_size);

// Each task writes to its slice without copying
for (i, slice) in slices.into_iter().enumerate() {
    spawn(async move {
        read_chunk_into(slice, i).await;
    });
}
```

**Benefits**:
- Eliminates memory copies during parallel loading
- 50% speedup over naive copy-based approach
- Safe wrapper over unsafe pointer manipulation

## Platform-Specific Behavior

The module automatically selects the best backend:

```rust
// On Linux: uses io_uring::load
// On macOS/Windows: uses async_io::load
let data = backends::load("file.bin").await?;
```

Manual backend selection when needed:

```rust
// Force specific backend
#[cfg(target_os = "linux")]
use tensor_store::backends::io_uring::load;

#[cfg(not(target_os = "linux"))]
use tensor_store::backends::async_io::load;
```

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

## Performance Tips

1. **Use parallel loading for large files** (>100MB)
2. **Match chunk count to CPU cores** for optimal parallelism
3. **Use buffer pool** for repeated allocations
4. **Consider O_DIRECT** for very large models
5. **Prefer mmap** for random access patterns
6. **Benchmark your specific workload** - optimal backend varies
