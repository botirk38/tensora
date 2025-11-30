# demo

Interactive demonstration tool showcasing SafeTensors and ServerlessLLM loader capabilities.

## Overview

This binary provides hands-on demonstrations of various tensor loading strategies for both SafeTensors and ServerlessLLM formats. It's designed to help users understand the different loading approaches and their characteristics.

## Usage

```bash
cargo run --bin demo -- <COMMAND> <SCENARIO> [OPTIONS]
```

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
