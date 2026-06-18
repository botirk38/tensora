# Tensora

**Adaptive checkpoint loading for large language models.**

Tensora is an open-source framework that applies workload-aware heuristics to select optimal I/O strategies for loading LLM checkpoints. It supports SafeTensors and ServerlessLLM storage layouts with pluggable I/O backends (synchronous POSIX, Tokio async, Linux `io_uring`, memory-mapped) and automatically picks the fastest path based on checkpoint size, shard structure, and platform capabilities.

> **Paper:** *Load by Design: Adaptive Heuristics for LLM Checkpoint Loading*
> — Botir Khaltaev (2026). Sources in [`paper/`](paper/).

---

## Key Results

| Regime | Winner | Mechanism |
|--------|--------|-----------|
| Small/single-shard SafeTensors | `sync` | Thread-parallel chunked POSIX reads |
| Large multi-shard SafeTensors (≥ 4 GB) | `io_uring` | Multi-worker ring submission |
| Range-heavy ServerlessLLM | `async` | Tokio grouped per-file tasks |
| Large partitioned ServerlessLLM | `io_uring` | Batched submission with coalescing |

The adaptive `default` I/O backend reproduces these crossovers automatically.

---

## Quick Start

```bash
# Build
cargo build --release

# Load a model (downloads from HuggingFace Hub on first run)
cargo run --release --bin profile -- safetensors default --model-id Qwen/Qwen3-0.6B --iterations 1

# Demo all I/O backends
cargo run --release --bin demo -- safetensors all --model-id Qwen/Qwen3-0.6B
```

### Python Bindings

```bash
cd bindings/python
uv sync --group dev --group torch
uv run python examples/pytorch.py gpt2 --prompt "Hello"
```

---

## Architecture

```
tensora/
├── src/
│   ├── storage/        # I/O backends (sync, tokio, io_uring, mmap)
│   ├── formats/        # Checkpoint formats (SafeTensors, ServerlessLLM)
│   ├── converters/     # Format conversion pipelines
│   ├── hf_model.rs     # HuggingFace Hub integration
│   └── bin/            # CLI tools (profile, demo, convert)
├── bindings/python/    # Python package (tensora_py) with PyTorch/vLLM support
├── benches/            # Criterion.rs benchmarks
├── scripts/            # Benchmark orchestration scripts
├── paper/              # LaTeX sources for the paper
└── results/            # Archived experiment data
```

---

## Requirements

| Component | Version |
|-----------|---------|
| Rust | ≥ 1.92 |
| OS | Linux (full feature set with `io_uring`); other platforms lack `io_uring` |
| Python | ≥ 3.12 (for bindings) |
| uv | Latest (for Python workflows) |

---

## Testing

```bash
cargo test --lib --locked
cargo clippy --lib --locked -- -D warnings
```

---

## Benchmarks

### Rust (Criterion)

```bash
export TENSORA_MODEL_ID=openai-community/gpt2
cargo bench
```

### Python (pytest-benchmark)

```bash
export TENSORA_BENCH_MODELS=openai-community/gpt2
./scripts/run_benchmarks.sh
```

---

## Reproducing Paper Experiments

See the paper's [Experimental Setup](paper/sections/experimental_setup.tex) for cold-cache methodology. In brief:

```bash
# Drop caches (requires root)
sync && echo 3 > /proc/sys/vm/drop_caches

# Run profiling
cargo run --release --bin profile -- safetensors sync --model-id Qwen/Qwen3-8B --iterations 1
```

Replication targets **storage-engine ordering and regime behaviour** (e.g., SafeTensors crossover point, ServerlessLLM async advantage), not identical millisecond timings across hardware.

---

## Citation

```bibtex
@software{khaltaev2026tensora,
  title   = {Load by Design: Adaptive Heuristics for LLM Checkpoint Loading},
  author  = {Khaltaev, Botir},
  year    = {2026},
  url     = {https://github.com/botirk38/tensora},
  license = {Apache-2.0}
}
```

See also: [`CITATION.cff`](CITATION.cff) and [`CITATION.bib`](CITATION.bib).

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
