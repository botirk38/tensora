# Python Benchmarks

pytest-benchmark suites measuring end-to-end model loading performance.

## Suites

| File | Measures |
|------|----------|
| `bench_safetensors.py` | SafeTensors loading via `tensora_py` |
| `bench_serverlessllm.py` | ServerlessLLM loading via `tensora_py` |
| `bench_vllm.py` | vLLM engine startup with Tensora weight loader |

## Running

### Via orchestration script (recommended)

```bash
export TENSORA_BENCH_MODELS=openai-community/gpt2
./scripts/run_benchmarks.sh
```

### Manual (single suite)

```bash
cd bindings/python
uv sync --group dev --group vllm
uv run maturin develop --release
uv run pytest benchmarks/bench_safetensors.py -v --model-id openai-community/gpt2
```

## Output

JSON files via `--benchmark-json` flag, written to `results/benchmarks/` by the orchestration script.
