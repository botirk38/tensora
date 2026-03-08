"""ServerlessLLM format unit tests (smoke, error paths)."""

import pytest
import torch

from tensor_store_py import load_serverlessllm_sync, open_serverlessllm_sync
from tests.fixtures import write_serverlessllm_dir


# --- Smoke ---

def test_open_smoke(tmp_path):
    tensors = {"x": torch.randn(2, 3)}
    out_dir = write_serverlessllm_dir(tensors, tmp_path / "model")
    handle = open_serverlessllm_sync(str(out_dir))
    assert handle.keys() == ["x"]
    t = handle.get_tensor("x")
    assert isinstance(t, torch.Tensor)
    assert t.shape == (2, 3)


def test_load_file_smoke(tmp_path):
    tensors = {"a": torch.zeros(1, 2), "b": torch.ones(3, 4)}
    out_dir = write_serverlessllm_dir(tensors, tmp_path / "model")
    d = load_serverlessllm_sync(str(out_dir))
    assert "a" in d and "b" in d
    assert d["a"].shape == (1, 2)
    assert d["b"].shape == (3, 4)


# --- Error paths ---

def test_open_nonexistent_path():
    with pytest.raises(FileNotFoundError, match="path not found"):
        open_serverlessllm_sync("/nonexistent/path/model_dir")


def test_load_file_nonexistent_serverlessllm():
    with pytest.raises(FileNotFoundError, match="path not found"):
        load_serverlessllm_sync("/nonexistent/path/model_dir")


def test_open_missing_index(tmp_path):
    (tmp_path / "tensor.data_0").write_bytes(b"")
    with pytest.raises(Exception):
        open_serverlessllm_sync(str(tmp_path))


def test_get_tensor_nonexistent_key(serverlessllm_dir_small):
    handle = open_serverlessllm_sync(serverlessllm_dir_small)
    with pytest.raises(Exception):
        handle.get_tensor("nonexistent")
