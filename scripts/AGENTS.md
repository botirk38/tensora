# AGENTS.md — Scripts

## Purpose

Orchestration scripts for running Python benchmarks at scale.

## Key Scripts

- `run_benchmarks.sh` — Main entry point for pytest-benchmark runs
- `paper_pytest_ladder.sh` — Full paper reproduction sweep

## Conventions

- Scripts run from the **repository root**
- Environment variables for configuration (see `scripts/README.md`)
- JSON output to `results/benchmarks/`
- vLLM runs serialized (one process at a time); non-vLLM can parallelize

## Do NOT

- Run scripts from within `scripts/` directory
- Hard-code model paths (use `TENSORA_BENCH_MODELS`)
- Modify existing result JSON files
