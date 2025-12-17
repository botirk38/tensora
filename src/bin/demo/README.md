# demo

Interactive demonstration tool showcasing SafeTensors and ServerlessLLM loader capabilities.

## Overview

This binary provides hands-on demonstrations of various tensor loading strategies for both SafeTensors and ServerlessLLM formats. It's designed to help users understand the different loading approaches and their characteristics.

## Usage

```bash
cargo run [--release] --bin demo -- <COMMAND> <SCENARIO> [OPTIONS]
```

**Important**: The `--release` flag (optional but recommended) goes before `--bin`, and `--` separates cargo flags from program arguments.

## Commands

### SafeTensors Demonstrations

```bash
cargo run --bin demo -- safetensors <SCENARIO> [--fixture <NAME>]
```

**Available Scenarios:**
- `async` - Async loading with io_uring (Linux) or tokio (other platforms)
- `sync` - Synchronous loading
- `mmap` - Memory-mapped lazy loading
- `parallel` - Parallel multi-core loading
- `metadata` - Detailed tensor metadata exploration
- `all` - Run all scenarios sequentially

### ServerlessLLM Demonstrations

```bash
cargo run --bin demo -- serverlessllm <SCENARIO> [--fixture <NAME>]
```

**Available Scenarios:**
- `async` - Async loading with partition information
- `sync` - Synchronous loading
- `mmap` - Memory-mapped lazy loading
- `metadata` - Index structure and partition statistics
- `all` - Run all scenarios sequentially

## Options

- `-f, --fixture <NAME>` - Specify a fixture name (e.g., qwen2-0.5b, mistral-7b)

## Examples

Run async SafeTensors demo:
```bash
cargo run --bin demo -- safetensors async
```

Run all ServerlessLLM scenarios:
```bash
cargo run --bin demo -- serverlessllm all
```

Demo with specific fixture:
```bash
cargo run --bin demo -- safetensors parallel --fixture qwen2-0.5b
```

Explore ServerlessLLM metadata:
```bash
cargo run --bin demo -- serverlessllm metadata --fixture mistral-7b
```

## Purpose

Use this tool to:
- Understand different loading strategies
- Compare performance characteristics
- Explore tensor metadata structures
- Learn about partition-based loading in ServerlessLLM

## Sample Output

### SafeTensors Async Demo
```
=== SafeTensors Async Load Demo ===
Loading: fixtures/qwen-qwen2-0.5b/model.safetensors
Backend: io_uring (Linux)
Loaded 201 tensors in 52.3ms

Sample tensors:
  model.embed_tokens.weight: F16 [151936, 896] (272.5 MB)
  model.layers.0.mlp.down_proj.weight: F16 [896, 4864] (8.3 MB)
  ...
```

### ServerlessLLM Metadata Demo
```
=== ServerlessLLM Metadata Demo ===
Index: fixtures/qwen-qwen2-0.5b/model_serverlessllm/

Partitions: 8
Total tensors: 201
Total size: 494.03 MB

Partition distribution:
  Partition 0: 26 tensors (62.1 MB)
  Partition 1: 25 tensors (61.8 MB)
  ...
```

## Fixture Setup

Before running demos, download test fixtures:

```bash
cd scripts
uv run python download_models.py Qwen/Qwen2-0.5B --convert --verify
```

This creates:
```
fixtures/
└── qwen-qwen2-0.5b/
    ├── model.safetensors          # For SafeTensors demos
    └── model_serverlessllm/       # For ServerlessLLM demos
```

## Building for Release

For accurate performance characteristics, build in release mode:

```bash
cargo build --release --bin demo
./target/release/demo safetensors all --fixture qwen-qwen2-0.5b
```

## See Also

- [Benchmarks](../../../benches/README.md) - For rigorous performance measurement
- [Profile Binary](../profile/README.md) - For profiling with external tools
- [SafeTensors Module](../../safetensors/README.md)
- [ServerlessLLM Module](../../serverlessllm/README.md)
