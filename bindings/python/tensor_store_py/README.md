# tensor_store_py

Python bindings for the tensor_store Rust library.

## Modules

- **_tensor_store_rust** — Native extension (load/save functions, handles)
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
from tensor_store_py.torch import load_safetensors

state_dict = load_safetensors("model.safetensors", device="cuda")
```

See `examples/` for runnable inference scripts.