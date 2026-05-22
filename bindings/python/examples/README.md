# Examples

Runnable inference scripts demonstrating `tensora_py` usage.

## Setup

```bash
cd bindings/python
uv sync --group dev --group torch
uv run maturin develop --release
```

## PyTorch

```bash
uv run python examples/pytorch.py gpt2 --prompt "Hello, world!"
```

Options: `--backend` (default/sync/io-uring), `--format` (safetensors/serverlessllm), `--device` (cuda/cpu)

## vLLM

```bash
uv run python examples/vllm_infer.py --model gpt2 --prompt "Hello, world!"
```

Options: `--backend`, `--gpu-memory-utilization`, `--max-model-len`
