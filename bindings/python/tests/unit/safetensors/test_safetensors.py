"""SafeTensors format unit tests (smoke, error paths, dtype roundtrip)."""

import pytest

torch = pytest.importorskip("torch")

import tensora._tensora_rust as rust_ext

from tensora._tensora_rust import (
    load_safetensors,
    load_safetensors_async,
    load_safetensors_sync,
    open_safetensors,
)
from tests.fixtures import write_safetensors


# --- Smoke ---


def test_open_smoke(tmp_path):
    tensors = {"x": torch.randn(2, 3)}
    path = write_safetensors(tensors, tmp_path / "test.safetensors")
    with pytest.raises(Exception, match="expected a directory"):
        open_safetensors(str(path))


def test_open_directory_smoke(tmp_path):
    tensors = {"x": torch.randn(2, 3)}
    model_dir = tmp_path / "test_dir"
    model_dir.mkdir()
    write_safetensors(tensors, model_dir / "model.safetensors")
    f = open_safetensors(str(model_dir))
    assert f.keys() == ["x"]
    t = f.get_tensor("x")
    assert isinstance(t, torch.Tensor)
    assert t.shape == (2, 3)


def test_load_directory_smoke(tmp_path):
    tensors = {"a": torch.zeros(1, 2), "b": torch.ones(3, 4)}
    model_dir = tmp_path / "test_dir"
    model_dir.mkdir()
    write_safetensors(tensors, model_dir / "model.safetensors")
    d = load_safetensors(str(model_dir))
    assert "a" in d and "b" in d
    assert d["a"].shape == (1, 2)
    assert d["b"].shape == (3, 4)


@pytest.mark.skipif(not hasattr(rust_ext, "load_safetensors_io_uring"), reason="io_uring binding is Linux-only")
def test_load_directory_io_uring_smoke(tmp_path):
    load_safetensors_io_uring = rust_ext.load_safetensors_io_uring
    tensors = {"a": torch.zeros(1, 2), "b": torch.ones(3, 4)}
    model_dir = tmp_path / "test_dir_io_uring"
    model_dir.mkdir()
    write_safetensors(tensors, model_dir / "model.safetensors")
    d = load_safetensors_io_uring(str(model_dir))
    assert "a" in d and "b" in d
    assert d["a"].shape == (1, 2)
    assert d["b"].shape == (3, 4)


# --- Error paths ---


def test_open_nonexistent_path():
    with pytest.raises(FileNotFoundError, match="path not found"):
        open_safetensors("/nonexistent/path/model_dir")


def test_load_file_nonexistent_path():
    with pytest.raises(FileNotFoundError, match="path not found"):
        load_safetensors("/nonexistent/path/model_dir")


def test_load_file_path_rejected(tmp_path):
    tensors = {"x": torch.randn(2, 3)}
    path = write_safetensors(tensors, tmp_path / "test.safetensors")
    with pytest.raises(Exception, match="expected a directory"):
        load_safetensors(str(path))


# --- Dtype and value roundtrip ---


def test_dtype_roundtrip_f32(safetensors_path_dtypes):
    d = load_safetensors(safetensors_path_dtypes)
    assert d["f32"].dtype == torch.float32
    assert d["f32"].tolist() == [1.0, 2.0, 3.0]


def test_dtype_roundtrip_i64(safetensors_path_dtypes):
    d = load_safetensors(safetensors_path_dtypes)
    assert d["i64"].dtype == torch.int64
    assert d["i64"].tolist() == [1, 2, 3]


def test_dtype_roundtrip_bool(safetensors_path_dtypes):
    d = load_safetensors(safetensors_path_dtypes)
    assert d["bool"].dtype == torch.bool
    assert d["bool"].tolist() == [True, False, True]


def test_scalar_tensor(safetensors_path_dtypes):
    d = load_safetensors(safetensors_path_dtypes)
    assert d["scalar"].shape == ()
    assert d["scalar"].item() == 42.0


def test_empty_tensor(safetensors_path_dtypes):
    d = load_safetensors(safetensors_path_dtypes)
    assert d["empty"].shape == (0, 2)


def test_get_tensor_nonexistent_key(safetensors_path_small):
    handle = open_safetensors(safetensors_path_small)
    with pytest.raises(Exception):
        handle.get_tensor("nonexistent")
