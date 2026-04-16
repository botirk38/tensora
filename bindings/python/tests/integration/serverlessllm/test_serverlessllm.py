"""ServerlessLLM format integration tests (backend parity)."""

import pytest

torch = pytest.importorskip("torch")

from tensora._tensora_rust import (
    load_serverlessllm,
    load_serverlessllm_async,
    load_serverlessllm_sync,
    open_serverlessllm,
)


# --- Backend: default ---


def test_open_default(serverlessllm_dir, hidden_dim):
    handle = open_serverlessllm(serverlessllm_dir)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (1024, hidden_dim)


def test_load_serverlessllm_default(serverlessllm_dir, hidden_dim):
    d = load_serverlessllm(serverlessllm_dir)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (1024, hidden_dim)
    assert d["wpe"].shape == (128, hidden_dim)


# --- Backend: sync ---


def test_load_serverlessllm_sync(serverlessllm_dir, hidden_dim):
    d = load_serverlessllm_sync(serverlessllm_dir)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (1024, hidden_dim)
    assert d["wpe"].shape == (128, hidden_dim)


# --- Backend: async ---


def test_open_async(serverlessllm_dir, hidden_dim):
    handle = open_serverlessllm(serverlessllm_dir)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (1024, hidden_dim)


def test_load_serverlessllm_async(serverlessllm_dir, hidden_dim):
    d = load_serverlessllm_async(serverlessllm_dir)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (1024, hidden_dim)


# --- Backend parity (default/sync/mmap consistency) ---


def test_load_keys_parity(serverlessllm_dir):
    keys_default = set(load_serverlessllm(serverlessllm_dir).keys())
    keys_sync = set(load_serverlessllm_sync(serverlessllm_dir).keys())
    keys_async = set(load_serverlessllm_async(serverlessllm_dir).keys())
    assert keys_default == keys_sync == keys_async


def test_load_shapes_parity(serverlessllm_dir):
    d_default = load_serverlessllm(serverlessllm_dir)
    d_sync = load_serverlessllm_sync(serverlessllm_dir)
    d_async = load_serverlessllm_async(serverlessllm_dir)
    assert (
        {k: tuple(d_default[k].shape) for k in d_default}
        == {k: tuple(d_sync[k].shape) for k in d_sync}
        == {k: tuple(d_async[k].shape) for k in d_async}
    )


def test_open_handle_parity(serverlessllm_dir):
    h_default = open_serverlessllm(serverlessllm_dir)
    h_async = open_serverlessllm(serverlessllm_dir)
    assert set(h_default.keys()) == set(h_async.keys())
    for k in h_default.keys():
        assert h_default.get_tensor(k).shape == h_async.get_tensor(k).shape
        assert torch.equal(h_default.get_tensor(k), h_async.get_tensor(k))
