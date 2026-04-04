# TensorStore Examples

## Setup

```bash
cd bindings/python
uv sync --group dev --group torch       # PyTorch
uv sync --group dev --group vllm       # vLLM
uv run maturin develop --release
```

## PyTorch Example

```bash
uv run python examples/pytorch.py gpt2 --prompt "Hello, world!"
```

Options:
- `model`: HuggingFace model ID
- `--prompt`: Text prompt
- `--backend`: default, sync, io-uring
- `--format`: safetensors (default) or serverlessllm
- `--device`: cuda or cpu

## vLLM Example

```bash
uv run python examples/vllm_infer.py --model gpt2 --prompt "Hello, world!"
```

Options:
- `--model`: HuggingFace model ID
- `--prompt`: Text prompt
- `--backend`: default, sync, io-uring
- `--gpu-memory-utilization`: 0.7 (default)
- `--max-model-len`: auto-detected from model config

## Formats

- SafeTensors: default for PyTorch
- ServerlessLLM: default for vLLM, optional for PyTorch with `--format serverlessllm`