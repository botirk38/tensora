# tensor_store

High-performance tensor I/O library optimized for io_uring on Linux, with cross-platform support.

## Overview

`tensor_store` is a Rust library designed for efficient tensor file loading and storage, developed as part of a Final Year Project (CS3821) at Royal Holloway, University of London. The project investigates whether Linux's modern io_uring interface can provide significant performance improvements for Large Language Model tensor loading compared to traditional approaches.

### Key Features

- **Zero-copy I/O** with io_uring on Linux (kernel 5.1+)
- **Parallel loading** with batched io_uring operations
- **Cross-platform support** with Tokio fallback for non-Linux systems
- **Multiple format support**: SafeTensors, ServerlessLLM formats
- **Memory-mapped I/O** support via memmap2
- **O_DIRECT support** for bypassing page cache
- **Buffer pooling** with thread-local caching for reduced allocation overhead
- **Jemalloc allocator** for improved memory management

### Research Goal

Investigate whether io_uring can exceed multi-threaded performance for LLM model loading, with a target of >20% improvement over baseline implementations.

## Quick Demo

Want to see tensor_store in action? First, install a test fixture:

```bash
# Step 1: Install a test model fixture (REQUIRED - auto-converts to both formats)
cd scripts
uv sync
uv run python download_models.py Qwen/Qwen2-0.5B
cd ..

# Step 2: Run demos
cargo run --release --bin demo -- safetensors all
cargo run --release --bin demo -- serverlessllm all
```

The script automatically:
- Downloads SafeTensors from HuggingFace
- Converts to ServerlessLLM format with optimal partition count (4 partitions for ~500MB model)
- Builds the converter binary if needed

> **Note**: Without a fixture, you'll see: `Error: No fixtures found under 'fixtures/'`

See [demo README](src/bin/demo/README.md) for more options.

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tensor_store = "0.1.0"
```

### Basic Usage

```rust
use tensor_store::safetensors;

// Async loading (uses io_uring on Linux, Tokio elsewhere)
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tensors = safetensors::load("model.safetensors").await?;

    println!("Loaded {} tensors", tensors.names().len());
    for name in tensors.names() {
        let tensor = tensors.tensor(name).unwrap();
        println!("  {}: {:?} {:?}", name, tensor.shape(), tensor.dtype());
    }

    Ok(())
}
```

### Parallel Loading (Large Files)

```rust
use tensor_store::safetensors;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parallel loading with N chunks (good for files >100MB)
    let cores = num_cpus::get();
    let tensors = safetensors::load_parallel("model.safetensors", cores).await?;

    println!("Loaded {} tensors", tensors.tensors().names().len());
    Ok(())
}
```

### Synchronous Loading

```rust
use tensor_store::safetensors;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // No async runtime needed
    let tensors = safetensors::load_sync("model.safetensors")?;
    println!("Loaded {} tensors", tensors.names().len());
    Ok(())
}
```

### Memory-Mapped Loading

```rust
use tensor_store::safetensors;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Zero-copy, lazy loading via mmap
    let tensors = safetensors::load_mmap("model.safetensors")?;

    // Data loaded on access
    for name in tensors.tensors().names() {
        let tensor = tensors.tensors().tensor(name)?;
        let data = tensor.data(); // Page fault here
    }

    Ok(())
}
```

## Project Structure

```
tensor_store/
├── src/
│   ├── lib.rs                    # Public API
│   ├── backends/                 # Platform-specific I/O implementations
│   │   ├── io_uring.rs          # io_uring backend (Linux)
│   │   ├── async_io.rs          # Tokio async backend (cross-platform)
│   │   ├── sync_io.rs           # Synchronous std::fs backend
│   │   ├── mmap.rs              # Memory-mapped I/O
│   │   ├── odirect.rs           # O_DIRECT bypass (Linux)
│   │   ├── batch.rs             # Batched operations
│   │   └── buffer_slice.rs      # Zero-copy buffer abstractions
│   ├── safetensors/             # SafeTensors format support
│   │   ├── reader.rs            # SafeTensors reading
│   │   └── writer.rs            # SafeTensors writing
│   ├── serverlessllm/           # ServerlessLLM format support
│   │   ├── reader.rs            # Partitioned tensor reading
│   │   ├── writer.rs            # Partitioned tensor writing
│   │   └── types.rs             # Common types
│   ├── converters/              # Format conversion utilities
│   ├── types/                   # Common types, traits, errors
│   └── bin/                     # Binary applications
│       ├── convert/             # SafeTensors → ServerlessLLM converter
│       ├── demo/                # Interactive demonstration tool
│       └── profile/             # Performance profiling harness
├── benches/                     # Criterion benchmarks
├── profiling/                   # External profiling binaries
├── scripts/                     # Python utility scripts
└── fixtures/                    # Test model fixtures
```

## Supported Formats

### SafeTensors

The standard format from HuggingFace for safe tensor storage.

```rust
use tensor_store::safetensors;

// Load
let tensors = safetensors::load("model.safetensors").await?;

// Access tensors
let weight = tensors.tensor("model.weight").unwrap();
println!("Shape: {:?}, Dtype: {:?}", weight.shape(), weight.dtype());

// Write
let mut writer = safetensors::SafeTensorsWriter::new();
writer.add_tensor("weight", tensor_view)?;
writer.write_to_file("output.safetensors").await?;
```

See [src/safetensors/README.md](src/safetensors/README.md) for details.

### ServerlessLLM

Partitioned format for efficient parallel loading, based on the ServerlessLLM paper (OSDI '24).

```rust
use tensor_store::serverlessllm;

// Load partitioned model
let model = serverlessllm::load("model_serverlessllm").await?;
println!("Partitions: {}", model.num_partitions());

// Access tensors
for (name, tensor) in &model {
    println!("{}: {:?}", name, tensor.shape());
}
```

See [src/serverlessllm/README.md](src/serverlessllm/README.md) for details.

## Binary Applications

### convert

Convert SafeTensors to ServerlessLLM format:

```bash
cargo run --release --bin convert -- model.safetensors ./output 8
```

See [src/bin/convert/README.md](src/bin/convert/README.md) for details.

### demo

Interactive demonstration of loading strategies:

```bash
# SafeTensors demos
cargo run --release --bin demo -- safetensors async
cargo run --release --bin demo -- safetensors parallel
cargo run --release --bin demo -- safetensors mmap

# ServerlessLLM demos
cargo run --release --bin demo -- serverlessllm async
cargo run --release --bin demo -- serverlessllm metadata
```

See [src/bin/demo/README.md](src/bin/demo/README.md) for details.

### profile

Performance profiling without benchmark overhead:

```bash
# Profile different loaders
cargo run --release --bin profile -- safetensors io-uring-load
cargo run --release --bin profile -- safetensors tokio-load
cargo run --release --bin profile -- serverlessllm async-load

# With flamegraph
cargo flamegraph --bin profile -- safetensors io-uring-load
```

See [src/bin/profile/README.md](src/bin/profile/README.md) for details.

## Benchmarks

Run the Criterion benchmark suite:

```bash
# All benchmarks
cargo bench

# Specific format
cargo bench --bench safetensors
cargo bench --bench serverlessllm

# Specific backend
cargo bench -- io_uring
cargo bench -- sync
cargo bench -- mmap
```

See [benches/README.md](benches/README.md) for details.

## Performance

### Backend Comparison

| Backend | Platform | Typical Throughput | Best For |
|---------|----------|-------------------|----------|
| io_uring | Linux 5.1+ | 9-12 GB/s | Large files, production |
| Tokio | All | 8-9 GB/s | Cross-platform apps |
| Sync | All | 7-8 GB/s | CLI tools, scripts |
| mmap | All | Variable | Random access |

### Key Optimizations

- **Zero-copy parallel loading**: Pre-allocate final buffer, split into non-overlapping slices for parallel tasks. Eliminates memory copies, providing ~50% speedup.

- **Buffer pooling**: Thread-local buffer pool with O(1) allocation. Provides ~70% speedup over non-pooled allocation.

- **Batched io_uring**: Submit all read operations before awaiting completions. Maximizes kernel batching efficiency.

- **O_DIRECT support**: Bypass page cache for large, one-time reads. Avoids cache pollution.

### Reference Results

On Linux 6.17.0 with NVMe SSD (523MB SafeTensors file):

| Method | Time | Throughput |
|--------|------|------------|
| io_uring parallel (16 chunks) | ~52ms | 10.0 GB/s |
| io_uring sequential | ~55ms | 9.5 GB/s |
| Sync (std::fs) | ~58ms | 9.0 GB/s |
| Original safetensors crate | ~60ms | 8.7 GB/s |

## Platform Support

| Platform | Backend | io_uring | Parallel | mmap | O_DIRECT |
|----------|---------|----------|----------|------|----------|
| Linux 5.1+ | io_uring | ✅ | ✅ | ✅ | ✅ |
| Linux (older) | Tokio | ❌ | ✅ | ✅ | ❌ |
| macOS | Tokio | ❌ | ✅ | ✅ | ❌ |
| Windows | Tokio | ❌ | ✅ | ✅ | ❌ |

## Requirements

- **Rust**: 2024 edition (1.85+)
- **Linux**: Kernel 5.1+ for io_uring features
- **Dependencies**: See `Cargo.toml`

## Development

### Building

```bash
# Debug build
cargo build

# Release build (for benchmarking)
cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test safetensors
cargo test serverlessllm
cargo test backends
```

### Documentation

```bash
# Generate and open docs
cargo doc --open
```

### Linting

```bash
# Run clippy with strict settings
cargo clippy -- -D warnings
```

### Setting Up Test Fixtures

```bash
cd scripts
uv sync
uv run python download_models.py Qwen/Qwen2-0.5B
```

This automatically downloads and converts to both SafeTensors and ServerlessLLM formats.

## Documentation

| Document | Description |
|----------|-------------|
| [CHANGELOG.md](CHANGELOG.md) | Version history and changes |
| [diary.md](diary.md) | Development diary and reflections |
| [src/backends/README.md](src/backends/README.md) | I/O backend architecture |
| [src/safetensors/README.md](src/safetensors/README.md) | SafeTensors format support |
| [src/serverlessllm/README.md](src/serverlessllm/README.md) | ServerlessLLM format support |
| [src/converters/README.md](src/converters/README.md) | Format conversion utilities |
| [benches/README.md](benches/README.md) | Benchmark suite |
| [profiling/README.md](profiling/README.md) | Profiling guide |
| [scripts/README.md](scripts/README.md) | Python utility scripts |

## Academic Context

This library is part of a **Final Year Project (CS3821)** at Royal Holloway, University of London.

**Research Question**: Can io_uring's kernel-offloaded operations eliminate CPU overhead from thread context switching and synchronization, achieving measurably better performance than traditional multi-threaded approaches for LLM tensor loading?

**Key Findings**:
- io_uring provides ~5% improvement over Tokio for sequential reads
- Zero-copy optimizations (buffer slicing) provide ~50% improvement
- Buffer pooling provides ~70% improvement over non-pooled allocation
- The bottleneck shifted from I/O interface to memory allocation patterns

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

- [ServerlessLLM Paper (OSDI '24)](https://www.usenix.org/conference/osdi24/presentation/fu)
- [SafeTensors Format](https://github.com/huggingface/safetensors)
- [io_uring Documentation](https://kernel.dk/io_uring.pdf)
- [tokio-uring Crate](https://github.com/tokio-rs/tokio-uring)
