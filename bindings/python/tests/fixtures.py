"""Test fixtures: create_gpt2 and ServerlessLLM writer (pure Python, no external binaries)."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# dtype string used by tensor_store ServerlessLLM format
_DTYPE_MAP = {
    torch.float32: "torch.float32",
    torch.float16: "torch.float16",
    torch.bfloat16: "torch.bfloat16",
    torch.float64: "torch.float64",
    torch.int32: "torch.int32",
    torch.int64: "torch.int64",
    torch.int16: "torch.int16",
    torch.int8: "torch.int8",
    torch.uint8: "torch.uint8",
    torch.uint16: "torch.uint16",
    torch.uint32: "torch.uint32",
    torch.uint64: "torch.uint64",
    torch.bool: "torch.bool",
}


def create_gpt2(n_layers: int = 12) -> dict[str, torch.Tensor]:
    """Create GPT-2-like tensor dict (same structure as safetensors benches)."""
    tensors = {}
    tensors["wte"] = torch.zeros((50257, 768))
    tensors["wpe"] = torch.zeros((1024, 768))
    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = torch.zeros((768,))
        tensors[f"h.{i}.ln_1.bias"] = torch.zeros((768,))
        tensors[f"h.{i}.attn.bias"] = torch.zeros((1, 1, 1024, 1024))
        tensors[f"h.{i}.attn.c_attn.weight"] = torch.zeros((768, 2304))
        tensors[f"h.{i}.attn.c_attn.bias"] = torch.zeros((2304,))
        tensors[f"h.{i}.attn.c_proj.weight"] = torch.zeros((768, 768))
        tensors[f"h.{i}.attn.c_proj.bias"] = torch.zeros((768,))
        tensors[f"h.{i}.ln_2.weight"] = torch.zeros((768,))
        tensors[f"h.{i}.ln_2.bias"] = torch.zeros((768,))
        tensors[f"h.{i}.mlp.c_fc.weight"] = torch.zeros((768, 3072))
        tensors[f"h.{i}.mlp.c_fc.bias"] = torch.zeros((3072,))
        tensors[f"h.{i}.mlp.c_proj.weight"] = torch.zeros((3072, 768))
        tensors[f"h.{i}.mlp.c_proj.bias"] = torch.zeros((768,))
    tensors["ln_f.weight"] = torch.zeros((768,))
    tensors["ln_f.bias"] = torch.zeros((768,))
    return tensors


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major contiguous stride."""
    stride = []
    s = 1
    for dim in reversed(shape):
        stride.append(s)
        s *= dim
    return tuple(reversed(stride))


def write_serverlessllm_dir(tensors: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Write tensors to ServerlessLLM format (tensor_index.json + tensor.data_0)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {}
    offset = 0
    partition_data = bytearray()

    for name, t in tensors.items():
        t = t.contiguous()
        shape = list(t.shape)
        stride = list(_contiguous_stride(t.shape))
        dtype_str = _DTYPE_MAP.get(t.dtype)
        if dtype_str is None:
            raise ValueError(f"unsupported dtype: {t.dtype}")
        data = t.numpy().tobytes()
        size = len(data)
        partition_data.extend(data)
        index[name] = [offset, size, shape, stride, dtype_str, 0]
        offset += size

    (out_dir / "tensor_index.json").write_text(json.dumps(index))
    (out_dir / "tensor.data_0").write_bytes(partition_data)
    return out_dir


def write_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> Path:
    """Write tensors to SafeTensors file."""
    path = Path(path)
    save_file(tensors, path)
    return path
