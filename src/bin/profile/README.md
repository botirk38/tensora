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
