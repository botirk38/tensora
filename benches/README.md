# Benchmarks

Performance benchmark suite for tensor_store using [Criterion.rs](https://github.com/bheisler/criterion.rs).

## Overview

The benchmark suite compares different I/O backends and loading strategies for SafeTensors and ServerlessLLM formats:

- **SafeTensors benchmarks** (`benches/safetensors.rs`)
- **ServerlessLLM benchmarks** (`benches/serverlessllm.rs`)

## Running Benchmarks

### Run all benchmarks

```bash
cargo bench
```

### Run specific format benchmarks

```bash
# SafeTensors only
cargo bench --bench safetensors

# ServerlessLLM only
cargo bench --bench serverlessllm
```

### Run specific backend benchmarks

```bash
# Run only io_uring benchmarks (Linux only)
cargo bench -- io_uring

# Run only sync benchmarks
cargo bench -- sync

# Run only mmap benchmarks
cargo bench -- mmap
```

## Benchmark Suite

### SafeTensors Benchmarks

| Benchmark | Platform | Description |
|-----------|----------|-------------|
| `io_uring_safetensors_load_{model}` | Linux only | Async loading with io_uring backend |
| `io_uring_safetensors_parallel_{N}_{model}` | Linux only | Parallel loading with N chunks using io_uring |
| `tokio_safetensors_load_{model}` | Non-Linux | Async loading with Tokio backend |
| `tokio_safetensors_parallel_4_{model}` | Non-Linux | Parallel loading with 4 chunks using Tokio |
| `sync_safetensors_load_{model}` | All platforms | Synchronous loading using std::fs |
| `mmap_safetensors_load_{model}` | All platforms | Memory-mapped loading |
| `original_safetensors_load_{model}` | All platforms | Reference implementation using safetensors crate |

### ServerlessLLM Benchmarks

| Benchmark | Platform | Description |
|-----------|----------|-------------|
| `io_uring_serverlessllm_load_{model}` | Linux only | Async loading with io_uring backend |
| `tokio_serverlessllm_load_{model}` | Non-Linux | Async loading with Tokio backend |
| `sync_serverlessllm_load_{model}` | All platforms | Synchronous loading using std::fs |
| `mmap_serverlessllm_load_{model}` | All platforms | Memory-mapped loading |

## Test Fixtures

Benchmarks automatically discover model fixtures in the `fixtures/` directory:

### SafeTensors Format

```
fixtures/
└── {model-name}/
    ├── model.safetensors      # SafeTensors file
    └── README.md              # Model metadata
```

### ServerlessLLM Format

```
fixtures/
└── {model-name}/
    ├── model_serverlessllm/
    │   ├── tensor_index.json  # Metadata and tensor index
    │   ├── tensor.data_0      # Partitioned data files
    │   ├── tensor.data_1
    │   └── ...
    └── README.md
```

## Downloading Test Fixtures

Use the provided Python script to download models from HuggingFace:

```bash
cd scripts
uv run python download_models.py Qwen/Qwen2-0.5B --convert --verify
```

See [scripts/README.md](../scripts/README.md) for details.

## Understanding Results

### Criterion Output

Criterion provides statistical analysis including:
- Mean execution time
- Standard deviation
- Comparison with previous runs
- Outlier detection

Example output:
```
io_uring_safetensors_load_qwen-qwen2-0.5b
                        time:   [52.341 ms 52.523 ms 52.728 ms]
                        change: [-3.2415% -2.5741% -1.9251%] (p = 0.00 < 0.05)
                        Performance has improved.
```

### Interpreting Performance

**Cold vs Warm Cache**: First run may be slower due to cold file system cache. Criterion runs multiple iterations with warmup to measure warm cache performance.

**mmap Benchmarks**: The `touch_pages()` function ensures mmap benchmarks trigger page faults for realistic measurement. Without this, mmap would appear artificially fast by only mapping memory without actually reading data.

**Parallel Benchmarks**: Parallel loading with N chunks typically shows benefits for large files (>100MB) on systems with multiple cores and fast storage.

## Platform-Specific Benchmarks

### Linux (io_uring)

On Linux with kernel 5.1+, benchmarks use the io_uring backend:

```bash
# Run io_uring specific benchmarks
cargo bench -- io_uring
```

Features tested:
- Zero-copy I/O with io_uring
- Parallel batched operations
- Fixed buffer pools

### Non-Linux (Tokio)

On macOS and Windows, benchmarks use Tokio's async I/O:

```bash
# Run tokio specific benchmarks
cargo bench -- tokio
```

## Profiling Benchmarks

### Flamegraph Integration

Generate flamegraphs for specific benchmarks:

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Profile SafeTensors loading
cargo flamegraph --bench safetensors -- --bench sync_safetensors

# Profile ServerlessLLM loading
cargo flamegraph --bench serverlessllm -- --bench io_uring_serverlessllm

# Profile with sudo for perf access (Linux)
sudo cargo flamegraph --bench safetensors -- --bench io_uring_safetensors
```

See [profiling/README.md](../profiling/README.md) for detailed profiling guide.

### Using the profiling binaries

For more control, use the dedicated profiling binaries:

```bash
# SafeTensors profiling
cargo run --bin safetensors_reader --release -- fixtures/model/model.safetensors

# ServerlessLLM profiling
cargo run --bin serverlessllm_reader --release -- fixtures/model/model_serverlessllm
```

## Benchmark Methodology

### File Discovery

Benchmarks automatically discover fixtures in `fixtures/` directory:
- Scans all subdirectories
- For SafeTensors: looks for `model.safetensors`
- For ServerlessLLM: looks for `model_serverlessllm/` directory
- Sorts by name for consistent ordering

### Measurement Strategy

Each benchmark:
1. Uses `criterion::black_box()` to prevent compiler optimizations
2. Measures tensor count and data size
3. For mmap: touches pages to trigger real I/O
4. Runs multiple iterations for statistical significance

### Performance Considerations

- **warmup**: Criterion performs warmup runs to populate file system cache
- **Outliers**: Criterion detects and reports statistical outliers
- **Noise**: System noise is measured and factored into confidence intervals

## Reference Results

Example results from development machine (Linux 6.17.0, 16 cores):

**SafeTensors (523MB file)**:
- `io_uring_parallel_16`: ~52ms
- `io_uring_load`: ~55ms
- `sync_safetensors`: ~58ms
- `mmap_safetensors`: ~65ms (with page touches)
- `original_safetensors`: ~60ms

**Key findings**:
- Parallel io_uring: ~5% faster than sync baseline
- Zero-copy optimizations: 50% faster than naive parallel
- Buffer pooling: 70% faster than non-pooled

## Troubleshooting

### "No benchmarks found"

Make sure test fixtures exist:
```bash
ls fixtures/*/model.safetensors
ls fixtures/*/model_serverlessllm/
```

### io_uring benchmarks not running

Check kernel version (need 5.1+):
```bash
uname -r
```

### Inconsistent results

- Close other applications to reduce system noise
- Run with `--sample-size 50` for more samples
- Check CPU frequency scaling settings
- Disable CPU turbo boost for consistency

### Permission errors with flamegraph

```bash
# Add perf_event_paranoid setting
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Or run with sudo
sudo -E cargo flamegraph --bench safetensors
```

## Contributing Performance Data

When reporting performance results:
1. Include system specs (CPU, RAM, storage type)
2. Include kernel version (Linux) or OS version
3. Specify fixture size and model name
4. Run with `--save-baseline` to track over time

```bash
# Save baseline for tracking
cargo bench --save-baseline my-machine

# Compare against baseline
cargo bench --baseline my-machine
```
