"""TensorFlow-specific convenience API for tensor_store."""

from typing import Dict, Optional, Union

from tensor_store_py._tensor_store_rust import (
    SafeTensorsHandlePy,
    load_safetensors,
    load_safetensors_mmap,
    load_safetensors_sync,
    save_safetensors,
    save_safetensors_bytes,
)

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TensorFlow is required for this module. Install with: pip install tensor-store-py[tensorflow]"
    )


def load_file(
    path: Union[str, "os.PathLike"], device: str = "/CPU:0"
) -> Dict[str, tf.Tensor]:
    """Load a safetensors file into a dict of TensorFlow tensors using default backend.

    Args:
        path: Path to the safetensors file.
        device: Device to load tensors to (e.g., "/CPU:0", "/GPU:0", "/device:CPU:0").

    Returns:
        Dict mapping tensor names to tf.Tensor values.
    """
    return load_safetensors(path, framework="tensorflow", device=device)


def load_file_mmap(
    path: Union[str, "os.PathLike"], device: str = "/CPU:0"
) -> Dict[str, tf.Tensor]:
    """Memory-map load a safetensors file into a dict of TensorFlow tensors.

    Args:
        path: Path to the safetensors file.
        device: Device to load tensors to (e.g., "/CPU:0", "/GPU:0").

    Returns:
        Dict mapping tensor names to tf.Tensor values.
    """
    return load_safetensors_mmap(path, framework="tensorflow", device=device)


def open_file(path: Union[str, "os.PathLike"]) -> SafeTensorsHandlePy:
    """Open a safetensors file for lazy loading.

    Args:
        path: Path to the safetensors file.

    Returns:
        SafeTensorsHandlePy for lazy tensor access.
    """
    return SafeTensorsHandlePy(path)


def save_file(
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
    save_safetensors(tensors, path, framework="tensorflow", metadata=metadata)


def save_file_bytes(
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
    result = save_safetensors_bytes(tensors, framework="tensorflow", metadata=metadata)
    if isinstance(result, bytes):
        return result
    raise TypeError("save_safetensors_bytes did not return bytes")


__all__ = [
    "load_file",
    "load_file_mmap",
    "open_file",
    "save_file",
    "save_file_bytes",
]
