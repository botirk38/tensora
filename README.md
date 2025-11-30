# tensor_store

High-performance tensor I/O library optimized for io_uring on Linux, with cross-platform support.

## Overview

`tensor_store` is a Rust library designed for efficient tensor file loading and storage, with a focus on leveraging modern I/O capabilities like io_uring for maximum performance on Linux systems.

### Key Features

- **Zero-copy I/O** with io_uring on Linux (kernel 5.1+)
- **Parallel loading** with batched io_uring operations
- **Cross-platform support** with Tokio fallback for non-Linux systems
- **Multiple format support**: SafeTensors, ServerlessLLM formats
- **Jemalloc allocator** for improved memory management
- **Memory-mapped I/O** support via memmap2
- **Fixed buffer pools** for kernel-registered buffers (eliminates copy overhead)

## Project Structure

```
tensor_store/
├── src/
│   ├── lib.rs              # Public API
│   ├── backends/           # Platform-specific I/O implementations
│   │   ├── iouring.rs      # io_uring backend (Linux only)
│   │   └── tokio.rs        # Tokio async backend (cross-platform)
│   ├── readers/            # Format readers
│   │   ├── safetensors.rs
│   │   └── serverlessllm.rs
│   ├── writers/            # Format writers
│   ├── converters/         # Format conversion
│   ├── types/              # Common types and traits
│   └── bin/                # Binary applications
│       ├── profile/        # Profiling tools
│       └── demo/           # Demo applications
├── benches/                # Performance benchmarks
│   ├── safetensors.rs
│   └── serverlessllm.rs
├── profiling/              # Additional profiling binaries
│   ├── safetensors_reader.rs
│   └── serverlessllm_reader.rs
├── fixtures/               # Test fixtures
├── scripts/                # Utility scripts
└── research/               # Research documentation
    ├── 01_serverlessllm_analysis.md
    ├── 02_iouring_ecosystem.md
    └── 03_tensorstore_format_spec.md
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tensor_store = "0.1.0"

# For io_uring support on Linux
[target.'cfg(target_os = "linux")'.dependencies]
tokio-uring = "0.5.0"
```

## Usage

### Basic Loading

```rust
use tensor_store;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a SafeTensors file
    let data = tensor_store::load_safetensors("model.safetensors").await?;
    println!("Loaded {} bytes", data.len());
    Ok(())
}
```

### Parallel Loading with io_uring (Linux only)

```rust
#[tokio_uring::main]
async fn main() -> std::io::Result<()> {
    // Load with parallel chunks (default: 4)
    let data = tensor_store::load_safetensors_parallel("model.safetensors").await?;

    // Or customize chunk count
    let data = tensor_store::load_safetensors_parallel_with_chunks("model.safetensors", 8).await?;

    println!("Loaded {} bytes", data.len());
    Ok(())
}
```

### Fixed Buffer Loading (Zero-copy)

```rust
use tensor_store::FixedBufPool;

#[tokio_uring::main]
async fn main() -> std::io::Result<()> {
    // Create buffer pool with 4 x 64MB buffers
    let bufs: Vec<Vec<u8>> = (0..4)
        .map(|_| vec![0u8; 64 * 1024 * 1024])
        .collect();
    let buf_pool = FixedBufPool::new(bufs);
    buf_pool.register()?;

    // Load using fixed buffers (zero-copy)
    let data = tensor_store::load_safetensors_fixed("model.safetensors", &buf_pool).await?;
    println!("Loaded {} bytes", data.len());
    Ok(())
}
```

## Binary Applications

### safetensors_reader

Profile SafeTensors file loading:

```bash
cargo run --bin safetensors_reader --release -- path/to/model.safetensors
```

### serverlessllm_reader

Profile ServerlessLLM format loading:

```bash
cargo run --bin serverlessllm_reader --release -- path/to/model.bin
```

### profile

General profiling tool:

```bash
cargo run --bin profile --release
```

### demo

Demo application showcasing library features:

```bash
cargo run --bin demo --release
```

## Benchmarks

Run performance benchmarks:

```bash
cargo bench
```

Available benchmarks:
- `safetensors` - SafeTensors format loading benchmarks
- `serverlessllm` - ServerlessLLM format loading benchmarks

Compares:
- Synchronous `std::fs` loading
- io_uring basic loading
- io_uring parallel loading
- io_uring fixed buffer loading

## Performance Optimizations

Based on io_uring ecosystem research:

- ✅ **File handle reuse**: Single file open for all parallel reads
- ✅ **Batched operations**: Submit all read operations before awaiting
- ✅ **Explicit cleanup**: Proper async file closing
- ✅ **Fixed buffers**: Kernel-registered buffers via `FixedBufPool` for zero-copy I/O
- ✅ **Memory-mapped files**: Optional mmap support for read-only access
- ⚠️ **IOPOLL**: Requires O_DIRECT (not currently exposed by tokio-uring)

See [research/02_iouring_ecosystem.md](research/02_iouring_ecosystem.md) for detailed analysis.

## Platform Support

| Platform | Backend | Features |
|----------|---------|----------|
| Linux 5.1+ | io_uring | Zero-copy I/O, parallel loading, fixed buffers |
| Linux (older) | Tokio | Standard async I/O |
| macOS | Tokio | Standard async I/O |
| Windows | Tokio | Standard async I/O |

## Requirements

- Rust 2024 edition
- Linux kernel 5.1+ for io_uring features
- Tokio runtime for non-Linux platforms

## Development

### Testing

```bash
cargo test
```

### Documentation

```bash
cargo doc --open
```

### CI/CD

This project uses GitLab CI with the [Rust CI templates](https://gitlab.com/rust-ci/rust-ci) for automated testing across multiple platforms.

## Academic Context

This library is part of a Final Year Project (CS3821) at Royal Holloway, University of London, investigating whether io_uring can exceed multi-threaded performance for LLM model loading.

**Research Goal**: Achieve >20% performance improvement over baseline multi-threaded implementations.

See documentation in `research/` for detailed analysis and findings.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

This is an academic project. For questions or suggestions, please open an issue.

## Related Resources

- [ServerlessLLM Analysis](research/01_serverlessllm_analysis.md)
- [io_uring Ecosystem Analysis](research/02_iouring_ecosystem.md)
- [TensorStore Format Specification](research/03_tensorstore_format_spec.md)
- [Project Roadmap](roadmap.md)
- [Development Diary](diary.md)
