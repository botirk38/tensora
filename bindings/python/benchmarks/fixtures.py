"""Benchmark-specific fixtures: create GPT-2-like tensors and write model formats."""

import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from tensor_store_py._tensor_store_rust import convert_safetensors_to_serverlessllm


def create_gpt2(n_layers: int = 2) -> dict[str, torch.Tensor]:
    """Create GPT-2-like tensor dict for benchmarks (small version)."""
    hidden_dim = 126  # must be divisible by 3 for split tests
    intermediate_dim = hidden_dim * 3

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
        str(safetensors_path),
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


def touch_tensor(t: torch.Tensor) -> float:
    """Force read of all pages (e.g. for mmap-backed tensors). Returns sum for sanity."""
    return t.sum().item()


def drop_page_cache(path: str | Path) -> None:
    """Hint kernel to drop page cache for path. Unix-only; no-op on Windows."""
    path = Path(path)
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        if path.is_file():
            with open(path, "rb") as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        else:
            for f in path.glob("*.data_*"):
                with open(f, "rb") as fp:
                    os.posix_fadvise(fp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def repo_dir_name(repo_id: str) -> str:
    """Convert a HuggingFace repo ID into a filesystem-friendly directory name."""
    return repo_id.replace("/", "-").lower()


def download_model_if_missing(model_name: str, fixtures_dir: Path) -> str:
    """Download model from HuggingFace if not already cached.

    Args:
        model_name: HuggingFace model ID (e.g., 'gpt2', 'Qwen/Qwen2-0.5B')
        fixtures_dir: Directory to store downloaded models

    Returns:
        Path to the downloaded SafeTensors file
    """
    from huggingface_hub import snapshot_download

    fixtures_dir = Path(fixtures_dir)
    model_dir = fixtures_dir / repo_dir_name(model_name)
    safetensors_path = model_dir / "model.safetensors"

    if safetensors_path.exists():
        return str(safetensors_path)

    print(f"Downloading {model_name} to {model_dir}...")

    download_dir = snapshot_download(
        repo_id=model_name,
        local_dir=str(model_dir),
        allow_patterns=["*.safetensors"],
        ignore_patterns=[
            "*.bin",
            "*.msgpack",
            "*.h5",
            "*.pb",
            "*.onnx",
            "*.tflite",
        ],
    )

    safetensors_files = list(Path(download_dir).glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {download_dir}")

    if len(safetensors_files) > 1:
        main_file = max(safetensors_files, key=lambda f: f.stat().st_size)
    else:
        main_file = safetensors_files[0]

    if main_file.name != "model.safetensors":
        import shutil

        shutil.copy2(main_file, safetensors_path)
    else:
        safetensors_path = main_file

    print(f"Downloaded {model_name} to {safetensors_path}")
    return str(safetensors_path)
