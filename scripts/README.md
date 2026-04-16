# Scripts

## Entry point

- **`run_benchmarks.sh`** — from the **repository root**, runs all pytest benchmarks under `bindings/python/benchmarks/` via `uv`, builds the extension with `maturin`, and writes one pytest-benchmark JSON file per model under `results/benchmarks/` (`pytest_benchmark_<slug>.json`). Requires **`TENSORA_BENCH_MODELS`** (one or more Hugging Face model ids, space- or comma-separated). With vLLM benchmarks, runs are serialized to a single pytest process; with `TENSORA_BENCH_NO_VLLM=1`, use **`TENSORA_BENCH_JOBS`** to cap parallel pytest runs.

## Typical usage

Run Python benchmarks:

```bash
export TENSORA_BENCH_MODELS=Qwen/Qwen3-8B
./scripts/run_benchmarks.sh
```

Multiple models (CPU-only stack; parallel by default up to four at a time):

```bash
export TENSORA_BENCH_MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3-8B"
export TENSORA_BENCH_NO_VLLM=1
./scripts/run_benchmarks.sh
```

Paper six-model ladder (SafeTensors + ServerlessLLM + vLLM; long-running, needs GPU for vLLM):

```bash
./scripts/paper_pytest_ladder.sh
```

Python dependencies: `cd bindings/python && uv sync` (see [`bindings/python/README.md`](../bindings/python/README.md)).

## Notes

- Rust-layer timings use the `profile` binary with `--model-id` or `--fixture`; see the repository root [`README.md`](../README.md).
- Archived TSV/JSON under `results/h100/` were produced on a dedicated experiment host; replicate **ordering and regime behaviour**, not byte-identical paths.
