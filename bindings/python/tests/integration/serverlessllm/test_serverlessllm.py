"""ServerlessLLM format integration tests (backend, parity)."""

import pytest
import torch

from tensor_store_py import (
    load_serverlessllm,
    load_serverlessllm_mmap,
    load_serverlessllm_sync,
    open_serverlessllm,
    open_serverlessllm_mmap,
    open_serverlessllm_sync,
)


# --- Backend: sync ---

def test_open_sync(serverlessllm_dir):
    handle = open_serverlessllm_sync(serverlessllm_dir)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (50257, 768)


def test_load_serverlessllm_sync(serverlessllm_dir):
    d = load_serverlessllm_sync(serverlessllm_dir)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (50257, 768)
    assert d["wpe"].shape == (1024, 768)


# --- Backend: mmap ---

def test_open_mmap(serverlessllm_dir):
    handle = open_serverlessllm_mmap(serverlessllm_dir)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (50257, 768)


def test_load_serverlessllm_mmap(serverlessllm_dir):
    d = load_serverlessllm_mmap(serverlessllm_dir)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (50257, 768)


# --- Backend: async ---

@pytest.mark.asyncio
async def test_open_async(serverlessllm_dir):
    handle = await open_serverlessllm(serverlessllm_dir)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (50257, 768)


@pytest.mark.asyncio
async def test_load_serverlessllm_async(serverlessllm_dir):
    d = await load_serverlessllm(serverlessllm_dir)
    assert d["wte"].shape == (50257, 768)
    assert d["wpe"].shape == (1024, 768)


# --- Backend parity (sync/mmap/async consistency) ---

def test_load_keys_parity(serverlessllm_dir):
    keys_sync = set(load_serverlessllm_sync(serverlessllm_dir).keys())
    keys_mmap = set(load_serverlessllm_mmap(serverlessllm_dir).keys())
    assert keys_sync == keys_mmap


@pytest.mark.asyncio
async def test_load_keys_parity_async(serverlessllm_dir):
    keys_sync = set(load_serverlessllm_sync(serverlessllm_dir).keys())
    keys_async = set((await load_serverlessllm(serverlessllm_dir)).keys())
    assert keys_sync == keys_async


def test_load_shapes_parity(serverlessllm_dir):
    d_sync = load_serverlessllm_sync(serverlessllm_dir)
    d_mmap = load_serverlessllm_mmap(serverlessllm_dir)
    assert {k: tuple(d_sync[k].shape) for k in d_sync} == {k: tuple(d_mmap[k].shape) for k in d_mmap}


@pytest.mark.asyncio
async def test_load_shapes_parity_async(serverlessllm_dir):
    d_sync = load_serverlessllm_sync(serverlessllm_dir)
    d_async = await load_serverlessllm(serverlessllm_dir)
    assert {k: tuple(d_sync[k].shape) for k in d_sync} == {k: tuple(d_async[k].shape) for k in d_async}


def test_open_handle_parity(serverlessllm_dir):
    h_sync = open_serverlessllm_sync(serverlessllm_dir)
    h_mmap = open_serverlessllm_mmap(serverlessllm_dir)
    assert set(h_sync.keys()) == set(h_mmap.keys())
    for k in h_sync.keys():
        assert h_sync.get_tensor(k).shape == h_mmap.get_tensor(k).shape
        assert torch.equal(h_sync.get_tensor(k), h_mmap.get_tensor(k))


@pytest.mark.asyncio
async def test_open_handle_parity_async(serverlessllm_dir):
    h_sync = open_serverlessllm_sync(serverlessllm_dir)
    h_async = await open_serverlessllm(serverlessllm_dir)
    assert set(h_sync.keys()) == set(h_async.keys())
    for k in h_sync.keys():
        assert h_sync.get_tensor(k).shape == h_async.get_tensor(k).shape
        assert torch.equal(h_sync.get_tensor(k), h_async.get_tensor(k))
