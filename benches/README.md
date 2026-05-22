# Benchmarks

Performance benchmark suite for tensora using [Criterion.rs](https://github.com/bheisler/criterion.rs).

## Overview

The benchmark suite measures I/O backends, format loading, conversion pipelines, and tensor access patterns across all tensora modules:

| Target | File | What it measures |
|--------|------|-----------------|
| SafeTensors | `benches/safetensors.rs` | Full-model load (default/sync/async/io_uring/mmap), sequential & random tensor access, native baseline |
| ServerlessLLM | `benches/serverlessllm.rs` | Full-model load (all backends), tensor access patterns, mmap page-touch |
| Conversion | `benches/conversion.rs` | SafeTensors → ServerlessLLM pipeline (default/sync/async/io_uring) |
| Backends | `benches/backends.rs` | Raw I/O micro-benchmarks: single-shard load, batch load, range-batch, mmap open+touch |
| Shared | `benches/bench_util.rs` | Model resolution, throughput helpers, shard enumeration |

All benchmarks report **throughput (bytes/sec)** alongside wall-clock time, use I/O-tuned measurement windows (10 samples, 15s measurement, 3s warmup), and force data materialization so mmap doesn't appear artificially fast.

## Running Benchmarks

### Prerequisites

Set **`TENSORA_MODEL_ID`** to a Hugging Face model id:

```bash
export TENSORA_MODEL_ID=openai-community/gpt2
```

The harness calls `tensora::hf_model` to ensure SafeTensors shards exist in the Hub cache and (for ServerlessLLM/conversion benches) a converted layout under the OS cache.

### Run all benchmarks

```bash
cargo bench
```

### Run specific targets

```bash
cargo bench --bench safetensors      # SafeTensors format
cargo bench --bench serverlessllm    # ServerlessLLM format
cargo bench --bench conversion       # Conversion pipeline
cargo bench --bench backends         # Raw I/O backends
```

### Filter by backend

```bash
cargo bench -- sync           # Sync-only benchmarks
cargo bench -- async          # Async-only
cargo bench -- io_uring       # io_uring (Linux)
cargo bench -- mmap           # Mmap
cargo bench -- tensor_random  # Random tensor access only
```

## Benchmark Groups

### SafeTensors (`benches/safetensors.rs`)

| Group | Benchmark | Platform | Description |
|-------|-----------|----------|-------------|
| `safetensors_default` | `load/{model}` | All | Adaptive backend selection |
| `safetensors_sync` | `load/{model}` | All | Synchronous loading |
| `safetensors_async` | `load/{model}` | All | Tokio async loading |
| `safetensors_io_uring` | `load/{model}` | Linux | io_uring loading |
| `safetensors_mmap` | `load/{model}` | All | Mmap with full page touch |
| `safetensors_tensor_sequential` | `scan/{model}` | All | Sequential tensor iteration on pre-loaded model |
| `safetensors_tensor_random` | `lookup/{model}` | All | Reverse-order tensor lookup on pre-loaded model |
| `native_safetensors` | `{model}/{shard}` | All | Per-shard native `safetensors` crate baseline |

### ServerlessLLM (`benches/serverlessllm.rs`)

| Group | Benchmark | Platform | Description |
|-------|-----------|----------|-------------|
| `serverlessllm_default` | `load/{model}` | All | Adaptive backend selection |
| `serverlessllm_sync` | `load/{model}` | All | Synchronous loading |
| `serverlessllm_async` | `load/{model}` | All | Tokio async loading |
| `serverlessllm_io_uring` | `load/{model}` | Linux | io_uring loading |
| `serverlessllm_mmap` | `load/{model}` | All | Mmap with full page touch |
| `serverlessllm_tensor_sequential` | `scan/{model}` | All | Sequential tensor iteration |
| `serverlessllm_tensor_random` | `lookup/{model}` | All | Reverse-order tensor lookup |
| `serverlessllm_mmap_tensor_sequential` | `scan/{model}` | All | Sequential access over mmap model |

### Conversion (`benches/conversion.rs`)

| Group | Benchmark | Platform | Description |
|-------|-----------|----------|-------------|
| `conversion_default` | `convert/{model}` | All | Adaptive conversion pipeline |
| `conversion_sync` | `convert/{model}` | All | Synchronous conversion |
| `conversion_async` | `convert/{model}` | All | Tokio async conversion |
| `conversion_io_uring` | `convert/{model}` | Linux | io_uring conversion |

### Backends (`benches/backends.rs`)

| Group | Benchmark | Platform | Description |
|-------|-----------|----------|-------------|
| `backend_sync_load` | `load/{model}` | All | SyncReader single-shard load |
| `backend_async_load` | `load/{model}` | All | AsyncReader single-shard load |
| `backend_io_uring_load` | `load/{model}` | Linux | io_uring Reader single-shard load |
| `backend_sync_batch` | `load_batch/{model}` | All | SyncReader all-shard batch load |
| `backend_async_batch` | `load_batch/{model}` | All | AsyncReader all-shard batch load |
| `backend_io_uring_batch` | `load_batch/{model}` | Linux | io_uring all-shard batch load |
| `backend_sync_range_batch` | `range_batch/{model}` | All | SyncReader per-tensor range reads |
| `backend_async_range_batch` | `range_batch/{model}` | All | AsyncReader per-tensor range reads |
| `backend_io_uring_range_batch` | `range_batch/{model}` | Linux | io_uring per-tensor range reads |
| `backend_mmap_open_touch` | `open_touch/{model}` | All | Mmap open + 4K page-walk |

## Understanding Results

### Criterion Output

Criterion provides statistical analysis including:
- Mean execution time and throughput (bytes/sec)
- Standard deviation
- Comparison with previous runs
- Outlier detection

Example output:
```
safetensors_sync/load/qwen-qwen3-0.6b
                        time:   [52.341 ms 52.523 ms 52.728 ms]
                        thrpt:  [9.4619 GiB/s 9.4946 GiB/s 9.5276 GiB/s]
                        change: [-3.2415% -2.5741% -1.9251%] (p = 0.00 < 0.05)
                        Performance has improved.
```

### Interpreting Performance

**Cold vs Warm Cache**: First run may be slower due to cold file system cache. Criterion runs multiple iterations with warmup to measure warm cache performance.

**mmap Benchmarks**: All mmap benchmarks touch every page (first and last byte of each tensor, or 4K page-walk for the backend mmap bench) to trigger real I/O. Without this, mmap would appear artificially fast by only mapping memory without reading data.

**Throughput**: Reported as bytes/sec based on total model or shard size, making comparisons meaningful across different model sizes.

## Measurement Configuration

I/O-heavy benchmarks use:
- `sample_size(10)` — fewer iterations to keep total bench time reasonable for multi-GB models
- `measurement_time(15s)` — longer per-iteration window for stable I/O measurements
- `warm_up_time(3s)` — sufficient warmup to populate page cache

Tensor access benchmarks use Criterion defaults (appropriate since the model is pre-loaded into memory).

## Platform-Specific Benchmarks

### Linux (io_uring)

On Linux with kernel 5.1+, io_uring benchmarks are included automatically:

```bash
cargo bench -- io_uring
```

### Non-Linux

On macOS and Windows, io_uring groups are excluded at compile time. All other benchmarks run normally.

## Profiling

### Flamegraph Integration

```bash
cargo install flamegraph
cargo flamegraph --bench safetensors -- safetensors_sync
sudo cargo flamegraph --bench backends -- backend_io_uring_load
```

See [profiling/README.md](../profiling/README.md) for detailed profiling guide.

### Using the `profile` binary

```bash
cargo run --release --bin profile -- safetensors sync --model-id openai-community/gpt2
cargo run --release --bin profile -- serverlessllm sync --model-id openai-community/gpt2
```

## Troubleshooting

### "Set TENSORA_MODEL_ID" / early exit

```bash
export TENSORA_MODEL_ID=openai-community/gpt2
cargo bench --bench safetensors
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
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
sudo -E cargo flamegraph --bench safetensors
```

## Contributing Performance Data

When reporting performance results:
1. Include system specs (CPU, RAM, storage type)
2. Include kernel version (Linux) or OS version
3. Specify model name and size
4. Run with `--save-baseline` to track over time

```bash
cargo bench --save-baseline my-machine
cargo bench --baseline my-machine
```
