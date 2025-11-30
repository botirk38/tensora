# Tensor Store Profiling Suite

This directory contains profiling harnesses for detailed performance analysis of tensor store readers. Unlike the benchmark suite in `../benches/`, these tools are optimized for deep profiling with tools like flamegraph, perf, and valgrind.

## Quick Start

### 1. Install Profiling Tools

```bash
# Flamegraph (recommended for visualization)
cargo install flamegraph

# Perf (Linux system profiler)
# Usually pre-installed on Linux distributions

# Valgrind (memory and call profiling)
# Install via package manager (apt install valgrind, etc.)
```

### 2. Prepare Test Data

#### For Safetensors Profiling
```bash
# Create a test safetensors file (use a real model for meaningful results)
# The harness expects: test_model.safetensors
cp /path/to/large/model.safetensors test_model.safetensors
```

#### For ServerlessLLM Profiling
```bash
# Create a test serverlessllm directory (use a real converted model)
# The harness expects: test_model_serverlessllm/
cp -r /path/to/converted/model test_model_serverlessllm
```

### 3. Run Profiling

```bash
# Profile safetensors reader with flamegraph
cargo flamegraph --bin safetensors_reader -- --profile io_uring

# Profile serverlessllm reader with perf
perf record -g cargo run --release --bin serverlessllm_reader -- --profile sync
perf report
```

## Profiling Modes

### Safetensors Reader (`safetensors_reader`)

**Available Modes:**
- `async` - Async I/O (uses io_uring on Linux, tokio elsewhere)
- `sync` - Synchronous loading (baseline)

**Example Commands:**
```bash
# Flamegraph profiling
cargo flamegraph --bin safetensors_reader -- --profile async
cargo flamegraph --bin safetensors_reader -- --profile sync

# Perf profiling
perf record -g cargo run --release --bin safetensors_reader -- --profile async
perf report

# Valgrind callgrind
valgrind --tool=callgrind cargo run --release --bin safetensors_reader -- --profile sync
kcachegrind callgrind.out.*
```

### ServerlessLLM Reader (`serverlessllm_reader`)

**Available Modes:**
- `async` - Async I/O (uses io_uring on Linux, tokio elsewhere)
- `sync` - Synchronous loading (baseline)

**Example Commands:**
```bash
# Flamegraph profiling
cargo flamegraph --bin serverlessllm_reader -- --profile async
cargo flamegraph --bin serverlessllm_reader -- --profile sync

# Perf profiling
perf record -g cargo run --release --bin serverlessllm_reader -- --profile async
perf report

# Valgrind callgrind
valgrind --tool=callgrind cargo run --release --bin serverlessllm_reader -- --profile sync
kcachegrind callgrind.out.*
```

## Advanced Profiling Techniques

### Memory Profiling with Valgrind

```bash
# Memory leak detection
valgrind --leak-check=full cargo run --release --bin safetensors_reader -- --profile sync

# Cache profiling
valgrind --tool=cachegrind cargo run --release --bin serverlessllm_reader -- --profile io_uring
cg_annotate cachegrind.out.*
```

### Custom Flamegraph Options

```bash
# Profile with higher frequency for more detail
CARGO_PROFILE_RELEASE_DEBUG=true flamegraph --freq 1000 --bin safetensors_reader -- --profile io_uring

# Profile only user-space code (exclude kernel)
flamegraph --no-kernel --bin serverlessllm_reader -- --profile tokio
```

### Perf Advanced Usage

```bash
# Profile with call graph and source line info
perf record -g --call-graph dwarf cargo run --release --bin safetensors_reader -- --profile io_uring

# Profile specific events
perf record -e cache-misses -g cargo run --release --bin serverlessllm_reader -- --profile sync

# Generate flamegraph from perf data
perf script | stackcollapse-perf.pl | flamegraph.pl > perf_flamegraph.svg
```

## Understanding Results

### Flamegraph Interpretation

- **Width**: Time spent in function (wider = more time)
- **Stack Depth**: Call hierarchy (deeper = more nested calls)
- **Colors**: Random (for visual distinction)

**Key Areas to Focus:**
- Wide bars at top = hot functions
- Deep stacks = complex call paths
- I/O operations (file reading, network)
- Memory allocations/deallocations

### Perf Report Analysis

- **Overhead %**: CPU time spent in function
- **Samples**: Number of profiling samples
- **Call Graph**: Function call relationships

**Common Bottlenecks:**
- `read()` / `write()` system calls
- Memory allocation functions (`malloc`, `free`)
- Serialization/deserialization code

### Profiling Best Practices

1. **Use Release Builds**: `cargo run --release` for realistic performance
2. **Warmup Iterations**: Profiling includes warmup to focus on hot paths
3. **Multiple Runs**: Run profiling multiple times for consistency
4. **Large Test Data**: Use realistic model sizes for meaningful results
5. **Compare Modes**: Profile different I/O modes to understand trade-offs

## Troubleshooting

### Common Issues

**"Test file/directory not found"**
- Ensure test data is in the correct location
- Run from the `tensor_store/` directory

**"Cannot allocate memory" when using flamegraph/perf**
- Profiling tools consume memory that conflicts with io_uring allocation
- Try running without profiling tools first: `cargo run --release --bin safetensors_reader -- --profile async`
- If using perf, increase memory limits: `sudo sysctl kernel.perf_event_mlock_kb=2048`
- Use sync mode as fallback: `cargo flamegraph --bin safetensors_reader -- --profile sync`
- Profiling binaries use reduced io_uring queue sizes (64 entries) to minimize memory usage

**"io_uring only available on Linux"**
- io_uring profiling requires Linux
- Use `tokio` or `sync` modes on other platforms

**"Permission denied" for perf**
- Run with `sudo` or configure perf permissions
- See: `echo 0 | sudo tee /proc/sys/kernel/kptr_restrict`

**Flamegraph shows mostly kernel time**
- Use `--no-kernel` flag to focus on user-space code
- Profile I/O heavy workloads may show kernel time

### Performance Tips

- **Large Models**: Use models >1GB for realistic profiling
- **SSD Storage**: Profile on SSD for I/O bound workloads
- **RAM**: Ensure sufficient RAM to avoid swapping
- **CPU**: Multi-core systems show parallel loading benefits

## Integration with Development Workflow

### CI/CD Profiling

```yaml
# Example GitHub Actions profiling step
- name: Profile Performance
  run: |
    cd tensor_store
    cargo flamegraph --bin safetensors_reader -- --profile io_uring --output profile.svg
    # Upload profile.svg as artifact
```

### Automated Regression Detection

```bash
# Compare profiling results over time
cargo flamegraph --bin serverlessllm_reader -- --profile sync --output baseline.svg
# After changes...
cargo flamegraph --bin serverlessllm_reader -- --profile sync --output current.svg
# Manually compare flamegraphs
```

## Contributing

When adding new profiling harnesses:

1. Follow the existing pattern with `--profile <mode>` argument
2. Include warmup iterations for cache population
3. Add comprehensive documentation
4. Test on multiple platforms (Linux/macOS/Windows)
5. Update this README with new harness information