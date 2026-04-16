"""PyTorch-specific convenience API for tensora."""

from typing import Dict, Optional, Union

from tensora._tensora_rust import (
    SafeTensorsHandlePy,
    load_safetensors as _load_safetensors,
    load_safetensors_async as _load_safetensors_async,
    load_safetensors_sync as _load_safetensors_sync,
    open_safetensors as _open_safetensors,
    save_safetensors as _save_safetensors,
    save_safetensors_bytes as _save_safetensors_bytes,
)

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for this module. Install with: pip install tensor-store-py[torch]"
    )


def load_safetensors(
    path: Union[str, "os.PathLike"], device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Load a safetensors model directory into a dict of PyTorch tensors.

    Args:
        path: Path to the safetensors model directory.
        device: Device to load tensors to (e.g., "cpu", "cuda:0").

    Returns:
        Dict mapping tensor names to torch.Tensor values.
    """
    return _load_safetensors(path, framework="torch", device=device)


def load_safetensors_async(
    path: Union[str, "os.PathLike"], device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Load a safetensors model directory asynchronously into PyTorch tensors."""
    return _load_safetensors_async(path, framework="torch", device=device)


def load_safetensors_sync(
    path: Union[str, "os.PathLike"], device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Load a safetensors model directory synchronously into PyTorch tensors."""
    return _load_safetensors_sync(path, framework="torch", device=device)


def open_safetensors(path: Union[str, "os.PathLike"]) -> SafeTensorsHandlePy:
    """Open a safetensors model directory for lazy loading.

    Args:
        path: Path to the safetensors model directory.

    Returns:
        SafeTensorsHandlePy for lazy tensor access.
    """
    return _open_safetensors(path)


def save_safetensors(
    tensors: Dict[str, torch.Tensor],
    path: Union[str, "os.PathLike"],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Save a dict of PyTorch tensors to a safetensors file.

    Args:
        tensors: Dict mapping tensor names to torch.Tensor values.
        path: Path to save the safetensors file.
        metadata: Optional metadata to include in the file header.
    """
    _save_safetensors(tensors, path, framework="torch", metadata=metadata)


def save_safetensors_bytes(
    tensors: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
) -> bytes:
    """Save a dict of PyTorch tensors to bytes in safetensors format.

    Args:
        tensors: Dict mapping tensor names to torch.Tensor values.
        metadata: Optional metadata to include in the file header.

    Returns:
        bytes: The safetensors file content as bytes.
    """
    result = _save_safetensors_bytes(tensors, framework="torch", metadata=metadata)
    if isinstance(result, bytes):
        return result
    raise TypeError("save_safetensors_bytes did not return bytes")


__all__ = [
    "load_safetensors",
    "load_safetensors_async",
    "load_safetensors_sync",
    "open_safetensors",
    "save_safetensors",
    "save_safetensors_bytes",
]
