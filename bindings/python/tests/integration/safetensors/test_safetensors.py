"""SafeTensors format integration tests (backend parity)."""

import pytest

torch = pytest.importorskip("torch")

from tensor_store_py._tensor_store_rust import (
    load_safetensors,
    load_safetensors_mmap,
    load_safetensors_sync,
    open_safetensors,
    open_safetensors_mmap,
    open_safetensors_sync,
)


# --- Backend: default ---


def test_open_default(safetensors_path, hidden_dim):
    handle = open_safetensors(safetensors_path)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (1024, hidden_dim)


def test_load_safetensors_default(safetensors_path, hidden_dim):
    d = load_safetensors(safetensors_path)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (1024, hidden_dim)


# --- Backend: sync ---


def test_open_sync(safetensors_path, hidden_dim):
    handle = open_safetensors_sync(safetensors_path)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (1024, hidden_dim)


def test_load_safetensors_sync(safetensors_path, hidden_dim):
    d = load_safetensors_sync(safetensors_path)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (1024, hidden_dim)


# --- Backend: mmap ---


def test_open_mmap(safetensors_path, hidden_dim):
    handle = open_safetensors_mmap(safetensors_path)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (1024, hidden_dim)


def test_load_safetensors_mmap(safetensors_path, hidden_dim):
    d = load_safetensors_mmap(safetensors_path)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (1024, hidden_dim)


# --- Backend parity (default/sync/mmap consistency) ---


def test_load_keys_parity(safetensors_path):
    keys_default = set(load_safetensors(safetensors_path).keys())
    keys_sync = set(load_safetensors_sync(safetensors_path).keys())
    keys_mmap = set(load_safetensors_mmap(safetensors_path).keys())
    assert keys_default == keys_sync == keys_mmap


def test_load_shapes_parity(safetensors_path):
    d_default = load_safetensors(safetensors_path)
    d_sync = load_safetensors_sync(safetensors_path)
    d_mmap = load_safetensors_mmap(safetensors_path)
    assert (
        {k: tuple(d_default[k].shape) for k in d_default}
        == {k: tuple(d_sync[k].shape) for k in d_sync}
        == {k: tuple(d_mmap[k].shape) for k in d_mmap}
    )


def test_open_handle_parity(safetensors_path):
    h_default = open_safetensors(safetensors_path)
    h_sync = open_safetensors_sync(safetensors_path)
    h_mmap = open_safetensors_mmap(safetensors_path)
    assert set(h_default.keys()) == set(h_sync.keys()) == set(h_mmap.keys())
    for k in h_default.keys():
        assert h_default.get_tensor(k).shape == h_sync.get_tensor(k).shape
        assert torch.equal(h_default.get_tensor(k), h_sync.get_tensor(k))
        assert h_default.get_tensor(k).shape == h_mmap.get_tensor(k).shape
        assert torch.equal(h_default.get_tensor(k), h_mmap.get_tensor(k))
