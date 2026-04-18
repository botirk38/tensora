"""SafeTensors format integration tests (backend parity)."""

import pytest

torch = pytest.importorskip("torch")

from tensora._tensora_rust import (
    load_safetensors,
    open_safetensors,
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


def test_load_safetensors_sync(safetensors_path, hidden_dim):
    d = load_safetensors(safetensors_path, backend="sync")
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (1024, hidden_dim)


# --- Backend: async ---


def test_load_safetensors_async(safetensors_path, hidden_dim):
    d = load_safetensors(safetensors_path, backend="async")
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (1024, hidden_dim)


def test_open_handle(safetensors_path, hidden_dim):
    handle = open_safetensors(safetensors_path)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (1024, hidden_dim)


# --- Backend parity (default/sync/async consistency) ---


def test_load_keys_parity(safetensors_path):
    keys_default = set(load_safetensors(safetensors_path).keys())
    keys_sync = set(load_safetensors(safetensors_path, backend="sync").keys())
    keys_async = set(load_safetensors(safetensors_path, backend="async").keys())
    assert keys_default == keys_sync == keys_async


def test_load_shapes_parity(safetensors_path):
    d_default = load_safetensors(safetensors_path)
    d_sync = load_safetensors(safetensors_path, backend="sync")
    d_async = load_safetensors(safetensors_path, backend="async")
    assert (
        {k: tuple(d_default[k].shape) for k in d_default}
        == {k: tuple(d_sync[k].shape) for k in d_sync}
        == {k: tuple(d_async[k].shape) for k in d_async}
    )


def test_open_handle_parity(safetensors_path):
    h_default = open_safetensors(safetensors_path)
    assert set(h_default.keys()) == set(open_safetensors(safetensors_path).keys())
    for k in h_default.keys():
        assert (
            h_default.get_tensor(k).shape
            == open_safetensors(safetensors_path).get_tensor(k).shape
        )
        assert torch.equal(
            h_default.get_tensor(k), open_safetensors(safetensors_path).get_tensor(k)
        )
