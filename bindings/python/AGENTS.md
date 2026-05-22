# AGENTS.md — Python Bindings

## Package

`tensora_py` — Python ≥ 3.12, built with maturin, managed with uv.

## Setup

```bash
cd bindings/python
uv sync --group dev --group torch
uv run maturin develop --release
```

## Structure

- `tensora/` — Python package source
- `examples/` — Runnable inference scripts
- `benchmarks/` — pytest-benchmark suites
- `tests/` — Unit and integration tests

## Conventions

- Native extension is `_tensora_rust` (built by maturin from workspace root)
- Framework APIs (`torch`, `tensorflow`) wrap the native extension
- Benchmarks use `--model-id` pytest option for model selection
- vLLM integration via custom weight loader class

## Testing

```bash
uv run pytest tests/ -v
```

## Do NOT

- Import PyTorch/TensorFlow at module top level (lazy import for optional deps)
- Modify benchmark JSON outputs under `results/`
- Use pip directly (use uv)
