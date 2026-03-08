"""API contract and export regression tests."""

import asyncio
import pytest

from tensor_store_py import (
    __version__,
    SafeTensorsHandle,
    ServerlessLLMHandle,
    TensorStoreError,
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


def test_all_exports_available():
    from tensor_store_py import __all__
    for name in __all__:
        obj = getattr(__import__("tensor_store_py"), name)
        assert obj is not None, f"{name} should be exported"


def test_handle_classes_are_types():
    assert isinstance(SafeTensorsHandle, type)
    assert isinstance(ServerlessLLMHandle, type)


def test_tensor_store_error_is_exception():
    assert issubclass(TensorStoreError, Exception)


@pytest.mark.asyncio
async def test_open_functions_return_awaitables(tmp_path):
    import torch
    safetensors_path = write_safetensors({"x": torch.zeros(1)}, tmp_path / "t.safetensors")
    serverless_path = write_serverlessllm_dir({"x": torch.zeros(1)}, tmp_path / "s")
    for fn, path in [
        (open_safetensors, str(safetensors_path)),
        (open_serverlessllm, str(serverless_path)),
        (load_safetensors, str(safetensors_path)),
    ]:
        assert callable(fn)
        result = fn(path)
        assert asyncio.iscoroutine(result) or hasattr(result, "__await__"), f"{fn.__name__} should return awaitable"


def test_sync_and_mmap_functions_are_callable():
    for fn in (
        open_safetensors_mmap,
        open_safetensors_sync,
        open_serverlessllm_mmap,
        open_serverlessllm_sync,
        load_safetensors_mmap,
        load_safetensors_sync,
        load_serverlessllm_mmap,
        load_serverlessllm_sync,
    ):
        assert callable(fn)
