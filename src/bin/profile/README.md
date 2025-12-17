# profile

Profiling harness for measuring tensor_store loader performance without Criterion overhead.

## Overview

This binary provides a lightweight profiling tool for benchmarking different tensor loading strategies. It's designed to work with profiling tools like `perf` and offers fine-grained control over profiling scenarios.

## Usage

```bash
cargo run --bin profile -- <COMMAND> <CASE> [OPTIONS]
```

## Commands

### SafeTensors Profiling

```bash
cargo run --bin profile -- safetensors <CASE> [--fixture <NAME>] [--iterations <N>]
```

**Available Cases:**
- `io-uring-load` - io_uring async load (Linux only)
- `io-uring-parallel` - io_uring parallel load (Linux only)
- `io-uring-prewarmed` - io_uring prewarmed load (Linux only)
- `tokio-load` - Tokio async load
- `tokio-parallel` - Tokio parallel load
- `tokio-prewarmed` - Tokio prewarmed load
- `sync` - Synchronous load
- `mmap` - Memory-mapped load
- `original` - Original safetensors crate load

### ServerlessLLM Profiling

```bash
cargo run --bin profile -- serverlessllm <CASE> [--fixture <NAME>] [--iterations <N>]
```

**Available Cases:**
- `async-load` - Async load
- `sync-load` - Synchronous load
- `mmap-load` - Memory-mapped load

## Options

- `-f, --fixture <NAME>` - Specify a fixture name (e.g., qwen2-0.5b, mistral-7b)
- `-i, --iterations <N>` - Number of iterations to run (default: 1)

## Examples

Profile io_uring async load:
```bash
cargo run --bin profile -- safetensors io-uring-load
```

Profile with perf:
```bash
cargo build --release --bin profile
perf record -g target/release/profile safetensors io-uring-load --fixture qwen2-0.5b
perf report
```

Profile with multiple iterations:
```bash
cargo run --bin profile -- serverlessllm async-load --iterations 10
```

Compare different loaders:
```bash
cargo run --release --bin profile -- safetensors original --fixture mistral-7b
cargo run --release --bin profile -- safetensors io-uring-load --fixture mistral-7b
```

## Purpose

Use this tool for:
- Performance profiling with external tools (perf, flamegraph)
- Comparing loader implementation performance
- Identifying performance bottlenecks
- Measuring impact of optimizations
- Generating profiling data without benchmark framework overhead

## Flamegraph Generation

Generate visual flamegraphs to identify hotspots:

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph for io_uring load
cargo flamegraph --bin profile -- safetensors io-uring-load --fixture qwen-qwen2-0.5b

# Output: flamegraph.svg (open in browser)
```

For more detailed instructions, see [profiling/README.md](../../../profiling/README.md).

## Interpreting Results

### Sample Output
```
=== SafeTensors io_uring Load Profile ===
Fixture: fixtures/qwen-qwen2-0.5b/model.safetensors
File size: 494.03 MB
Backend: io_uring

Iteration 1: 52.341ms (9.44 GB/s)
Iteration 2: 51.892ms (9.52 GB/s)
Iteration 3: 52.103ms (9.48 GB/s)

Average: 52.112ms (9.48 GB/s)
Std Dev: 0.225ms
```

### Performance Baseline Reference

| Loader | Typical Speed | Notes |
|--------|---------------|-------|
| `io-uring-load` | 9-10 GB/s | Best on Linux NVMe |
| `io-uring-parallel` | 10-12 GB/s | With 8+ cores |
| `tokio-load` | 8-9 GB/s | Cross-platform |
| `sync` | 7-8 GB/s | Baseline |
| `mmap` | Variable | Depends on access pattern |
| `original` | 6-7 GB/s | safetensors crate reference |

## Advanced Profiling

### With perf stat (CPU counters)
```bash
cargo build --release --bin profile
perf stat ./target/release/profile safetensors io-uring-load --iterations 10
```

### With perf record (sampling)
```bash
perf record -g ./target/release/profile safetensors io-uring-load --iterations 5
perf report
```

### With Valgrind (memory analysis)
```bash
valgrind --tool=cachegrind ./target/release/profile safetensors sync --iterations 3
```

## Fixture Setup

Download test fixtures before profiling:

```bash
cd scripts
uv run python download_models.py Qwen/Qwen2-0.5B --convert --verify
```

## See Also

- [Profiling Suite](../../../profiling/README.md) - Comprehensive profiling guide
- [Benchmarks](../../../benches/README.md) - Criterion-based benchmarks
- [Demo Binary](../demo/README.md) - Interactive demonstrations
