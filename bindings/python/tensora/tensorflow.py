"""TensorFlow-specific convenience API for tensora."""

from typing import Dict, Optional, Union

from tensora._tensora_rust import (
    SafeTensorsHandlePy,
    load_safetensors as _load_safetensors,
    iter_safetensors as _iter_safetensors,
    open_safetensors as _open_safetensors,
    save_safetensors as _save_safetensors,
    save_safetensors_bytes as _save_safetensors_bytes,
)

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TensorFlow is required for this module. Install with: pip install tensor-store-py[tensorflow]"
    )


def load_safetensors(
    path: Union[str, "os.PathLike"],
    backend: str = "default",
    device: str = "/CPU:0",
) -> Dict[str, tf.Tensor]:
        """Load a safetensors model directory into a dict of TensorFlow tensors.

    Args:
        path: Path to the safetensors model directory.
        backend: Backend to use ("default", "sync", "io_uring").
        device: Device to load tensors to (e.g., "/CPU:0", "/GPU:0").

    Returns:
        Dict mapping tensor names to tf.Tensor values.
    """
    return _load_safetensors(
        path, framework="tensorflow", device=device, backend=backend
    )


def iter_safetensors(
    path: Union[str, "os.PathLike"],
    backend: str = "default",
    device: str = "/CPU:0",
):
    """Iterate over a safetensors model directory as (name, tensor) pairs.

    Args:
        path: Path to the safetensors model directory.
        backend: Backend to use ("default", "sync", "io_uring").
        device: Device to load tensors to (e.g., "/CPU:0", "/GPU:0").

    Returns:
        Iterator yielding (name, tf.Tensor) pairs.
    """
    return _iter_safetensors(
        path, framework="tensorflow", device=device, backend=backend
    )


def open_safetensors(path: Union[str, "os.PathLike"]) -> SafeTensorsHandlePy:
    """Open a safetensors model directory for lazy loading.

    Args:
        path: Path to the safetensors model directory.

    Returns:
        SafeTensorsHandlePy for lazy tensor access.
    """
    return _open_safetensors(path)


def save_safetensors(
    tensors: Dict[str, tf.Tensor],
    path: Union[str, "os.PathLike"],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Save a dict of TensorFlow tensors to a safetensors file.

    Args:
        tensors: Dict mapping tensor names to tf.Tensor values.
        path: Path to save the safetensors file.
        metadata: Optional metadata to include in the file header.
    """
    _save_safetensors(tensors, path, framework="tensorflow", metadata=metadata)


def save_safetensors_bytes(
    tensors: Dict[str, tf.Tensor],
    metadata: Optional[Dict[str, str]] = None,
) -> bytes:
    """Save a dict of TensorFlow tensors to bytes in safetensors format.

    Args:
        tensors: Dict mapping tensor names to tf.Tensor values.
        metadata: Optional metadata to include in the file header.

    Returns:
        bytes: The safetensors file content as bytes.
    """
    result = _save_safetensors_bytes(tensors, framework="tensorflow", metadata=metadata)
    if isinstance(result, bytes):
        return result
    raise TypeError("save_safetensors_bytes did not return bytes")


__all__ = [
    "load_safetensors",
    "iter_safetensors",
    "open_safetensors",
    "save_safetensors",
    "save_safetensors_bytes",
]
