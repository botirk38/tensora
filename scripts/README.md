# Scripts

Benchmark orchestration for Python-layer performance measurements.

## Entry Points

| Script | Purpose |
|--------|---------|
| `run_benchmarks.sh` | Run pytest-benchmark suites (SafeTensors, ServerlessLLM, vLLM) |
| `paper_pytest_ladder.sh` | Full six-model sweep for paper tables |

## Usage

```bash
export TENSORA_BENCH_MODELS=openai-community/gpt2
./scripts/run_benchmarks.sh
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TENSORA_BENCH_MODELS` | Yes | HuggingFace model IDs (space/comma-separated) |
| `TENSORA_BENCH_NO_VLLM` | No | Set `1` to skip vLLM benchmarks |
| `TENSORA_BENCH_JOBS` | No | Max parallel pytest processes (default: min(4, models)) |
| `TENSORA_BENCH_JSON` | No | Output directory for JSON results |
| `TENSORA_SKIP_MATURIN` | No | Set `1` to skip extension rebuild |

## Output

JSON files under `results/benchmarks/` (or `$TENSORA_BENCH_JSON`), one per model.
