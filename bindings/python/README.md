# tensora_py

Python bindings for the Tensora checkpoint loading library.

## Setup

```bash
cd bindings/python
uv sync --group dev --group torch
uv run maturin develop --release
```

## Usage

```python
from tensora_py.torch import load_safetensors

state_dict = load_safetensors("model.safetensors", device="cuda")
```

## Modules

| Module | Description |
|--------|-------------|
| `tensora._tensora_rust` | Native Rust extension (load/save handles) |
| `tensora.torch` | PyTorch convenience API |
| `tensora.tensorflow` | TensorFlow convenience API |

## Examples

```bash
uv run python examples/pytorch.py gpt2 --prompt "Hello"
uv run python examples/vllm_infer.py --model gpt2 --prompt "Hello"
```

See [`examples/README.md`](examples/README.md) for full options.

## Benchmarks

```bash
export TENSORA_BENCH_MODELS=openai-community/gpt2
cd ../..
./scripts/run_benchmarks.sh
```

See [`benchmarks/README.md`](benchmarks/README.md) for details.

## Requirements

- Python ≥ 3.12
- uv (package manager)
- maturin (for building the Rust extension)
