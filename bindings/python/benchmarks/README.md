# Tensor Store Benchmarks

Real-model benchmarks using `pytest-benchmark`. Results can be exported as **JSON** via pytest-benchmark’s `--benchmark-json` (wired in `scripts/run_benchmarks.sh`).

## Run all pytest benchmarks (recommended)

From the **repository root**:

```bash
export TENSORA_BENCH_MODELS=openai-community/gpt2   # required (one or more ids)
./scripts/run_benchmarks.sh
```

This runs `uv sync` (dev + vLLM groups), `maturin develop --release`, then pytest on `bench_safetensors.py`, `bench_serverlessllm.py`, and `bench_vllm.py`. JSON is written to **`results/benchmarks/pytest_benchmark_<slug>.json`** per model (slug: repo id with `/` → `-`, lowercased). Set **`TENSORA_BENCH_JSON`** to an **output directory** to place those files elsewhere. Each run uses a distinct **`--cache-dir`** under that directory’s `.cache/<slug>/` so parallel jobs do not collide.

### Environment variables

| Variable | Meaning |
|----------|---------|
| `TENSORA_BENCH_MODELS` | **Required.** One or more HuggingFace model ids (`pytest --model-id`), space- and/or comma-separated. |
| `TENSORA_BENCH_JSON` | Optional. **Directory** for pytest-benchmark JSON files (default: `results/benchmarks/` under repo root). Must not be a `.json` file path. |
| `TENSORA_BENCH_JOBS` | Optional. Max concurrent pytest processes when **`TENSORA_BENCH_NO_VLLM=1`** (default: `min(4, number of models)`). Ignored when vLLM benchmarks run (always one job). |
| `TENSORA_SKIP_MAURIN=1` | Skip `maturin develop --release` if the extension is already built. |
| `TENSORA_BENCH_NO_VLLM=1` | Run only SafeTensors + ServerlessLLM benchmarks (omit `bench_vllm.py`). Uses `uv sync --group dev --group torch` instead of the `vllm` group. |

## Manual pytest (single suite)

```bash
cd bindings/python
uv sync --group dev --group vllm
uv run maturin develop --release

uv run pytest benchmarks/bench_safetensors.py -v --model-id openai-community/gpt2
uv run pytest benchmarks/bench_serverlessllm.py -v --model-id openai-community/gpt2
uv run pytest benchmarks/bench_vllm.py -v --model-id openai-community/gpt2
```

Add `--benchmark-json=path.json` to capture machine-readable timings.

**Rust cold matrices and archived TSVs:** Large sweeps that produced `results/h100/profile/*.tsv` and `results/h100/vllm/*.tsv` were run on a dedicated host by invoking the `profile` binary (Rust) and pytest/vLLM benchmarks with the cold-cache procedure from the paper—not via a bundled shell matrix. Treat those paths as **archival**; reproduce **ordering and regime behaviour** on your machine.

## CLI Options (pytest)

| Option | Description |
|--------|-------------|
| `--model-id` | **Required.** HuggingFace model ID |
| `--cache-dir` | Directory for downloaded models and conversion cache (default: pytest temp dir) |

## Benchmark Suites

| File | Description |
|------|-------------|
| `bench_safetensors.py` | SafeTensors loading: native vs tensora (`sync`, `async`, `default`, `open_*`) |
| `bench_serverlessllm.py` | ServerlessLLM loading (`sync`, `async`, `default`, `open_*`) |
| `bench_vllm.py` | vLLM integration: init, TTFT, steady-state decode |

## SafeTensors Benchmarks

**Backends:**

- `native` - `safetensors.torch.load_file`
- `tensora sync` - `tensora.load_safetensors_sync`
- `tensora async` - `tensora.load_safetensors_async`
- `tensora default` - `tensora.load_safetensors`
- `tensora open` - `tensora.open_safetensors`

**Cache modes:** `warm`, `cold`

## ServerlessLLM Benchmarks

Uses the shared size-based heuristic for partition count: `max(1, ceil(total_bytes / 512 MiB))`.

**Backends:**

- `sync`
- `async`
- `default`
- `open`

**Cache modes:** `warm`, `cold`

## vLLM Benchmarks

The vLLM subprocess harness sets **`enforce_eager=True`** so runs do not depend on a full TorchInductor/Triton host toolchain (GCC plus CUDA headers). Relative loader comparisons are unchanged; absolute init times may differ from deployments that use CUDA graphs.

**Loaders:**

- `native` - default vLLM loader
- `ts_safetensors_*` / `ts_serverlessllm_*` - tensora loaders (see `vllm_runner.py`)

**Benchmark kinds:**

- `load_only` - model initialization time
- `ttft` - time to first token
- `steady_state_decode` - average decode time after warmup

## Partition heuristic

Default ServerlessLLM partition counts follow the Rust helper:

`max(1, ceil(total_bytes / 512 MiB))`

There is no artificial upper cap (beyond practical `usize` limits). Override with explicit conversion
arguments if you need a different layout.

## Default backend policy

- `open_*` defaults to `mmap`.
- `load_*` defaults to eager loading and chooses between internal Rust backends.
- On Linux, `default` may select `io_uring` internally when it wins for the workload.
- `load_*` does not auto-select `mmap`.

## Recommended Models (H100 Box)

For the H100 box with 180GB RAM, use this fixture ladder:

- Tiny: `openai-community/gpt2`
- Small: `Qwen/Qwen2.5-0.5B-Instruct`
- Medium-small: `Qwen/Qwen2.5-1.5B-Instruct`
- Medium: `Qwen/Qwen2.5-3B-Instruct`
- Large: `Qwen/Qwen2.5-7B-Instruct`
- XL: `Qwen/Qwen2.5-14B-Instruct`
- XXL: `Qwen/Qwen2.5-32B-Instruct`

All models have open weights, safetensors format, and vLLM compatibility.
