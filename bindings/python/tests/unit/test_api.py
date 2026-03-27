"""API contract and export regression tests."""

import importlib

import pytest

from tensor_store_py import __version__
from tensor_store_py._tensor_store_rust import (
    TensorStoreError,
    SafeTensorsHandlePy,
    ServerlessLLMHandlePy,
    load_safetensors,
    load_safetensors_mmap,
    load_safetensors_sync,
    load_serverlessllm,
    load_serverlessllm_mmap,
    load_serverlessllm_sync,
    open_safetensors,
    open_safetensors_mmap,
    open_safetensors_sync,
    open_serverlessllm,
    open_serverlessllm_mmap,
    open_serverlessllm_sync,
)
from tests.fixtures import write_safetensors, write_serverlessllm_dir


def test_version_is_string():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_package_version_matches_rust_extension():
    import tensor_store_py._tensor_store_rust as ext

    assert ext.__version__ == __version__


def test_package_all_exports_version_only():
    import tensor_store_py

    assert tensor_store_py.__all__ == ["__version__"]
    assert getattr(tensor_store_py, "__version__") == __version__


def test_handle_classes_are_types():
    assert isinstance(SafeTensorsHandlePy, type)
    assert isinstance(ServerlessLLMHandlePy, type)


def test_tensor_store_error_is_exception():
    assert issubclass(TensorStoreError, Exception)


def test_default_functions_work(tmp_path):
    torch = pytest.importorskip("torch")

    safetensors_path = write_safetensors(
        {"x": torch.zeros(1)}, tmp_path / "t.safetensors"
    )
    serverless_path = write_serverlessllm_dir({"x": torch.zeros(1)}, tmp_path / "s")

    # Test open functions (default backend)
    handle = open_safetensors(str(safetensors_path))
    assert isinstance(handle, SafeTensorsHandlePy)
    assert "x" in handle.keys()

    handle = open_serverlessllm(str(serverless_path))
    assert isinstance(handle, ServerlessLLMHandlePy)
    assert "x" in handle.keys()

    # Test load functions (default backend)
    result = load_safetensors(str(safetensors_path))
    assert isinstance(result, dict)
    assert "x" in result

    result = load_serverlessllm(str(serverless_path))
    assert isinstance(result, dict)
    assert "x" in result


def test_all_backend_functions_are_callable():
    for fn in (
        open_safetensors,
        open_safetensors_mmap,
        open_safetensors_sync,
        open_serverlessllm,
        open_serverlessllm_mmap,
        open_serverlessllm_sync,
        load_safetensors,
        load_safetensors_mmap,
        load_safetensors_sync,
        load_serverlessllm,
        load_serverlessllm_mmap,
        load_serverlessllm_sync,
    ):
        assert callable(fn)


def test_rust_extension_is_importable():
    mod = importlib.import_module("tensor_store_py._tensor_store_rust")
    assert hasattr(mod, "__version__")
    assert hasattr(mod, "load_safetensors_sync")
