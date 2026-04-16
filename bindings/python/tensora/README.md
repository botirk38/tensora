# tensora_py

Python bindings for the tensora Rust library.

## Modules

- **_tensora_rust** — Native extension (load/save functions, handles)
- **torch** — PyTorch convenience API
- **tensorflow** — TensorFlow convenience API

## Installation

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

See `examples/` for runnable inference scripts.