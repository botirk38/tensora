# tensor_store Python Bindings

PyTorch-first Python bindings for the [tensor_store](../README.md) Rust library.

## Installation

Requires Rust 1.92+ (`rustup update`) and Python 3.9+.

```bash
cd bindings/python
uv sync
```

## Quick Start

```python
import asyncio
import torch
from tensor_store_py import open_safetensors_sync, load_file_sync

# Sync backend: lazy load (reads file into memory)
f = open_safetensors_sync("model.safetensors")
for name in f.keys():
    tensor = f.get_tensor(name)
    print(name, tensor.shape, tensor.dtype)

# Sync backend: eager load into dict
state_dict = load_file_sync("model.safetensors", device="cpu")
```

Async (no suffix, returns awaitable):

```python
async def main():
    f = await open_safetensors("model.safetensors")
    state_dict = await load_file("model.safetensors", device="cpu")
asyncio.run(main())
```

## API

- `open_safetensors`, `open_safetensors_sync`, `open_safetensors_mmap` - Lazy load SafeTensors
- `open_serverlessllm`, `open_serverlessllm_sync`, `open_serverlessllm_mmap` - Lazy load ServerlessLLM
- `load_file`, `load_file_sync`, `load_file_mmap` - Eager load into dict
- `Handle.keys()` - List tensor names
- `Handle.get_tensor(name, device="cpu")` - Get a single tensor as `torch.Tensor`

## Requirements

- Python 3.9+
- PyTorch 2.1+
- Linux or macOS (ServerlessLLM mmap is Linux-only; owned fallback elsewhere)

## Backends

- **async** (no suffix): Native async I/O (Tokio; io_uring on Linux), returns awaitable
- **sync** (`_sync` suffix): Reads file into memory, blocking I/O
- **mmap** (`_mmap` suffix): Memory-mapped, zero-copy on CPU

GIL released during I/O and parsing.
