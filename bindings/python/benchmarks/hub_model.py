"""Hugging Face model resolution and ServerlessLLM cache helpers for benchmarks."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from tensora._tensora_rust import convert_safetensors_to_serverlessllm


_RECOMMENDED_PARTITION_TARGET_BYTES = 512 * 1024 * 1024


def recommended_partition_count(total_bytes: int) -> int:
    """Default ServerlessLLM partition count: ``max(1, ceil(total_bytes / 512 MiB))``.

    This matches ``tensora::formats::serverlessllm::recommended_partition_count`` in Rust.
    """
    if total_bytes <= 0:
        return 1
    return max(
        1,
        (total_bytes + _RECOMMENDED_PARTITION_TARGET_BYTES - 1)
        // _RECOMMENDED_PARTITION_TARGET_BYTES,
    )


def _cache_root() -> Path:
    return Path(__file__).parent / ".cache"


def repo_dir_name(repo_id: str) -> str:
    """Convert a HuggingFace repo ID into a filesystem-friendly directory name."""
    return repo_id.replace("/", "-").lower()


@dataclass
class ModelDescriptor:
    """Metadata about a real HuggingFace model for benchmarking."""

    model_id: str
    safetensors_files: list[Path]
    shard_count: int
    total_bytes: int
    partition_count: int


def get_model_safetensors(model_name: str, cache_dir: Path) -> list[Path]:
    """Download model from HuggingFace and return all safetensors shard paths.

    Args:
        model_name: HuggingFace model ID (e.g., 'gpt2', 'Qwen/Qwen2-0.5B')
        cache_dir: Directory to store downloaded models

    Returns:
        List of paths to all .safetensors shard files, sorted by name for determinism
    """
    from huggingface_hub import snapshot_download

    cache_dir = Path(cache_dir)
    model_dir = cache_dir / repo_dir_name(model_name)

    stale_artifact = model_dir / "model_serverlessllm"
    if stale_artifact.exists():
        import shutil

        shutil.rmtree(stale_artifact)

    if not model_dir.exists():
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
    else:
        download_dir = str(model_dir)

    safetensors_files = sorted(
        Path(download_dir).glob("*.safetensors"),
        key=lambda f: f.name,
    )

    if not safetensors_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {download_dir} for model {model_name}"
        )

    return safetensors_files


def get_model_descriptor(model_name: str, cache_dir: Path) -> ModelDescriptor:
    """Get full metadata for a real model.

    Returns a ModelDescriptor with file list, shard count, total size, and
    recommended partition count based on the size heuristic.
    """
    files = get_model_safetensors(model_name, cache_dir)
    total_bytes = sum(f.stat().st_size for f in files)
    parts = recommended_partition_count(total_bytes)

    return ModelDescriptor(
        model_id=model_name,
        safetensors_files=files,
        shard_count=len(files),
        total_bytes=total_bytes,
        partition_count=parts,
    )


def ensure_serverlessllm_artifact(
    model_name: str,
    safetensors_files: list[Path],
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Ensure a ServerlessLLM artifact exists for the model, converting if needed.

    Uses the source file path + revision + partition count as the cache key,
    so repeated runs with the same inputs reuse cached artifacts.

    Args:
        model_name: HuggingFace model ID
        safetensors_files: List of source safetensors shard paths
        revision: Optional revision (defaults to 'main')
        cache_dir: Optional base directory for ServerlessLLM conversion cache

    Returns:
        Path to the ServerlessLLM artifact directory
    """
    if cache_dir is None:
        cache_dir = _cache_root() / "serverlessllm"
    else:
        cache_dir = Path(cache_dir) / "serverlessllm"

    total_bytes = sum(f.stat().st_size for f in safetensors_files)
    partition_count_override = recommended_partition_count(total_bytes)

    source_dir = safetensors_files[0].parent
    shard_names = ":".join(f.name for f in sorted(safetensors_files))
    revision_str = revision or "main"

    key_input = f"{source_dir.resolve()}:{revision_str}:{partition_count_override}:{shard_names}:v3"
    cache_key = hashlib.sha256(key_input.encode()).hexdigest()[:16]
    out_dir = cache_dir / repo_dir_name(model_name) / cache_key

    index_path = out_dir / "tensor_index.json"
    if index_path.exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Converting {model_name} to ServerlessLLM "
        f"(partition_count={partition_count_override})..."
    )

    convert_safetensors_to_serverlessllm(
        str(source_dir),
        str(out_dir),
        partition_count_override,
    )

    return out_dir


def get_or_build_serverlessllm(
    model_name: str,
    cache_dir: Path,
) -> tuple[Path, ModelDescriptor]:
    """Convenience helper to get both the ServerlessLLM artifact and model metadata.

    Returns:
        Tuple of (serverlessllm_dir, model_descriptor)
    """
    desc = get_model_descriptor(model_name, cache_dir)

    sllm_dir = ensure_serverlessllm_artifact(
        model_name,
        desc.safetensors_files,
        cache_dir=cache_dir,
    )

    return sllm_dir, desc


def touch_tensor(t: torch.Tensor) -> float:
    """Force read of all pages (e.g. for mmap-backed tensors). Returns element count for sanity."""
    _ = t.sum()
    return float(t.numel())
