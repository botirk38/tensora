# Results

Cross-environment benchmark results for the Tensora checkpoint loading paper.

## Structure

Each environment directory (`h100/`, `h200/`, `a100/`) contains:

- `rust.tsv` — Rust-layer cold-cache tensor loading times (ms)
- `vllm.tsv` — vLLM integration benchmarks (H100: measured, A100: estimated from Rust-layer ratios)
- `anchor.tsv` — Five-repetition cold-cache anchor measurements (H100/H200 only)
- `env.txt` — Environment metadata (H200/A100)
- `raw/` — Raw pytest-benchmark JSON artifacts
- `analysis/` — Profiling notes (H100 only)

### TSV Formats

**rust.tsv** — `model, format, backend, time_ms, tensors, bytes`

**vllm.tsv** — `model, loader, kind, cache_mode, metric, value, status`

## Environments

| GPU | Storage | io_uring | Notes |
|-----|---------|----------|-------|
| H100 | Local NVMe | Working | Baseline |
| H200 | Network | Blocked (EPERM) | Selector adapts to async |
| A100 | Network | Blocked (EPERM) | vLLM metrics estimated |

## Reproducing

Rust profiling: `profile` binary with cold-cache procedure (see root `README.md`).
Python/vLLM: `scripts/run_benchmarks.sh` (see `bindings/python/benchmarks/README.md`).
