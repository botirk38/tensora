"""SafeTensors format unit tests (smoke, error paths, dtype roundtrip)."""

import pytest
import torch

from tensor_store_py import load_safetensors_sync, open_safetensors_sync
from tests.fixtures import write_safetensors


# --- Smoke ---

def test_open_smoke(tmp_path):
    tensors = {"x": torch.randn(2, 3)}
    path = write_safetensors(tensors, tmp_path / "test.safetensors")
    f = open_safetensors_sync(str(path))
    assert f.keys() == ["x"]
    t = f.get_tensor("x")
    assert isinstance(t, torch.Tensor)
    assert t.shape == (2, 3)


def test_load_file_smoke(tmp_path):
    tensors = {"a": torch.zeros(1, 2), "b": torch.ones(3, 4)}
    path = write_safetensors(tensors, tmp_path / "test.safetensors")
    d = load_safetensors_sync(str(path))
    assert "a" in d and "b" in d
    assert d["a"].shape == (1, 2)
    assert d["b"].shape == (3, 4)


# --- Error paths ---

def test_open_nonexistent_path():
    with pytest.raises(FileNotFoundError, match="path not found"):
        open_safetensors_sync("/nonexistent/path/model.safetensors")


def test_load_file_nonexistent_path():
    with pytest.raises(FileNotFoundError, match="path not found"):
        load_safetensors_sync("/nonexistent/path/model.safetensors")


# --- Dtype and value roundtrip ---

def test_dtype_roundtrip_f32(safetensors_path_dtypes):
    d = load_safetensors_sync(safetensors_path_dtypes)
    assert d["f32"].dtype == torch.float32
    assert d["f32"].tolist() == [1.0, 2.0, 3.0]


def test_dtype_roundtrip_i64(safetensors_path_dtypes):
    d = load_safetensors_sync(safetensors_path_dtypes)
    assert d["i64"].dtype == torch.int64
    assert d["i64"].tolist() == [1, 2, 3]


def test_dtype_roundtrip_bool(safetensors_path_dtypes):
    d = load_safetensors_sync(safetensors_path_dtypes)
    assert d["bool"].dtype == torch.bool
    assert d["bool"].tolist() == [True, False, True]


def test_scalar_tensor(safetensors_path_dtypes):
    d = load_safetensors_sync(safetensors_path_dtypes)
    assert d["scalar"].shape == ()
    assert d["scalar"].item() == 42.0


def test_empty_tensor(safetensors_path_dtypes):
    d = load_safetensors_sync(safetensors_path_dtypes)
    assert d["empty"].shape == (0, 2)


def test_get_tensor_nonexistent_key(safetensors_path_small):
    handle = open_safetensors_sync(safetensors_path_small)
    with pytest.raises(Exception):
        handle.get_tensor("nonexistent")
