"""Test fixtures: create_gpt2 and format writers."""

from pathlib import Path

import torch
from safetensors.torch import save_file
from tensora._tensora_rust import convert_safetensors_to_serverlessllm


def create_gpt2(n_layers: int = 2) -> dict[str, torch.Tensor]:
    """Create GPT-2-like tensor dict (same structure as safetensors benches, but small)."""
    hidden_dim = 126  # must be divisible by 3 for split tests
    intermediate_dim = hidden_dim * 3  # 378

    tensors = {}
    tensors["wte"] = torch.zeros((1024, hidden_dim))
    tensors["wpe"] = torch.zeros((128, hidden_dim))
    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = torch.zeros((hidden_dim,))
        tensors[f"h.{i}.ln_1.bias"] = torch.zeros((hidden_dim,))
        tensors[f"h.{i}.attn.bias"] = torch.zeros((1, 1, 128, 128))
        tensors[f"h.{i}.attn.c_attn.weight"] = torch.zeros(
            (hidden_dim, intermediate_dim)
        )
        tensors[f"h.{i}.attn.c_attn.bias"] = torch.zeros((intermediate_dim,))
        tensors[f"h.{i}.attn.c_proj.weight"] = torch.zeros((hidden_dim, hidden_dim))
        tensors[f"h.{i}.attn.c_proj.bias"] = torch.zeros((hidden_dim,))
        tensors[f"h.{i}.ln_2.weight"] = torch.zeros((hidden_dim,))
        tensors[f"h.{i}.ln_2.bias"] = torch.zeros((hidden_dim,))
        tensors[f"h.{i}.mlp.c_fc.weight"] = torch.zeros((hidden_dim, intermediate_dim))
        tensors[f"h.{i}.mlp.c_fc.bias"] = torch.zeros((intermediate_dim,))
        tensors[f"h.{i}.mlp.c_proj.weight"] = torch.zeros(
            (intermediate_dim, hidden_dim)
        )
        tensors[f"h.{i}.mlp.c_proj.bias"] = torch.zeros((hidden_dim,))
    tensors["ln_f.weight"] = torch.zeros((hidden_dim,))
    tensors["ln_f.bias"] = torch.zeros((hidden_dim,))
    return tensors


def write_serverlessllm_dir(
    tensors: dict[str, torch.Tensor], out_dir: Path, num_partitions: int = 1
) -> Path:
    """Write tensors to ServerlessLLM format via Rust bindings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = out_dir / "_source.safetensors"
    save_file(tensors, safetensors_path)
    convert_safetensors_to_serverlessllm(
        str(out_dir),
        str(out_dir),
        num_partitions,
    )
    safetensors_path.unlink()
    return out_dir


def write_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> Path:
    """Write tensors to SafeTensors file."""
    path = Path(path)
    save_file(tensors, path)
    return path


def write_safetensors_dir(tensors: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Write tensors to a one-file SafeTensors model directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out_dir / "model.safetensors")
    return out_dir
