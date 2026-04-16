# tensora Python Bindings

PyTorch-first Python bindings for the [tensora](../README.md) Rust library. The system is referred to as **Tensora** in the paper; package name remains `tensora_py`.

## Installation

Requires Rust 1.92+ (`rustup update`) and Python 3.12+.

### Core bindings

```bash
cd bindings/python
uv sync --group dev
uv run maturin develop --release
```

### PyTorch examples

```bash
uv sync --group dev --group torch
```

### TensorFlow examples

```bash
uv sync --group dev --group tensorflow
```

### vLLM examples and benchmarks

```bash
uv sync --group dev --group vllm
```

## Import layout

- **`tensora_py`** — package metadata only (`__version__` via `__all__`).
- **`tensora_py._tensora_rust`** — native I/O and handles (`load_*`, `open_*`, `save_*`, handle classes, `TensoraError`).
- **`tensora_py.torch`** — PyTorch-oriented helpers (`load_file`, `save_file`, …).
- **`tensora_py.tensorflow`** — TensorFlow-oriented helpers (`load_file`, `save_file`, …).

## Quick start (PyTorch)

```python
import asyncio

from tensora_py.torch import load_file as pytorch_load_file, open_file as pytorch_open_file

# Lazy: mmap-backed handle (see rust helpers for async/sync/mmap variants)
f = pytorch_open_file("model.safetensors")
for name in f.keys():
    tensor = f.get_tensor(name)
    print(name, tensor.shape, tensor.dtype)

# Eager: full dict load (sync path)
state_dict = pytorch_load_file("model.safetensors", device="cpu")
```

## Quick start (low-level extension)

```python
import asyncio

from tensora_py._tensora_rust import (
    load_safetensors_sync,
    open_safetensors,
    open_safetensors_sync,
)

# Blocking eager load (defaults to PyTorch tensors)
weights = load_safetensors_sync("model.safetensors")
```

Async entrypoints return awaitables (no `_sync` / `_mmap` suffix):

```python
async def main():
    from tensora_py._tensora_rust import load_safetensors, open_safetensors

    handle = await open_safetensors("model.safetensors")
    state = await load_safetensors("model.safetensors")

asyncio.run(main())
```

## TensorFlow

```python
from tensora_py.tensorflow import load_file as tf_load_file

weights = tf_load_file("model.safetensors", device="/CPU:0")
```

## Requirements

- Python 3.12+
- Linux or macOS (ServerlessLLM mmap is Linux-only; owned fallback elsewhere)

## Backends

- **default** (no suffix): eager load through the Rust selector; on Linux this may choose `io_uring`, `async`, or `sync`
- **async** (`_async` suffix): explicit Tokio async eager load
- **sync** (`_sync` suffix): explicit blocking eager load
- **open** (`open_*`): mmap-backed lazy handle when supported

The GIL is released during I/O and parsing where applicable.
