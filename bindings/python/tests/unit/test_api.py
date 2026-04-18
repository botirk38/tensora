"""API contract and export regression tests."""

import importlib
import platform

import pytest

from tensora import __version__
from tensora._tensora_rust import (
    TensoraError,
    SafeTensorsHandlePy,
    ServerlessLLMHandlePy,
    load_safetensors,
    load_serverlessllm,
    open_safetensors,
    open_serverlessllm,
)
from tests.fixtures import write_safetensors_dir, write_serverlessllm_dir


def test_version_is_string():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_package_version_matches_rust_extension():
    import tensora._tensora_rust as ext

    assert ext.__version__ == __version__


def test_package_all_exports_version_only():
    import tensora

    assert tensora.__all__ == ["__version__"]
    assert getattr(tensora, "__version__") == __version__


def test_handle_classes_are_types():
    assert isinstance(SafeTensorsHandlePy, type)
    assert isinstance(ServerlessLLMHandlePy, type)


def test_tensora_error_is_exception():
    assert issubclass(TensoraError, Exception)


def test_default_functions_work(tmp_path):
    torch = pytest.importorskip("torch")

    safetensors_path = write_safetensors_dir({"x": torch.zeros(1)}, tmp_path / "t")
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


def test_all_functions_are_callable():
    for fn in (
        open_safetensors,
        open_serverlessllm,
        load_safetensors,
        load_serverlessllm,
    ):
        assert callable(fn)


def test_rust_extension_is_importable():
    mod = importlib.import_module("tensora._tensora_rust")
    assert hasattr(mod, "__version__")
    assert hasattr(mod, "load_safetensors")


@pytest.mark.skipif(platform.system() != "Linux", reason="io_uring is Linux-only")
def test_load_with_io_uring_backend(tmp_path):
    torch = pytest.importorskip("torch")
    from tensora._tensora_rust import load_safetensors, load_serverlessllm

    safetensors_path = write_safetensors_dir({"x": torch.zeros(1)}, tmp_path / "t")
    result = load_safetensors(str(safetensors_path), backend="io_uring")
    assert isinstance(result, dict)
    assert "x" in result

    serverless_path = write_serverlessllm_dir({"x": torch.zeros(1)}, tmp_path / "s")
    result = load_serverlessllm(str(serverless_path), backend="io_uring")
    assert isinstance(result, dict)
    assert "x" in result
