"""SafeTensors format integration tests (backend, parity, async concurrency)."""

import asyncio
import pytest
import torch

from tensor_store_py import (
    load_safetensors,
    load_safetensors_mmap,
    load_safetensors_sync,
    open_safetensors,
    open_safetensors_mmap,
    open_safetensors_sync,
)


# --- Backend: sync ---

def test_open_sync(safetensors_path):
    handle = open_safetensors_sync(safetensors_path)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (50257, 768)


def test_load_safetensors_sync(safetensors_path):
    d = load_safetensors_sync(safetensors_path)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (50257, 768)


# --- Backend: mmap ---

def test_open_mmap(safetensors_path):
    handle = open_safetensors_mmap(safetensors_path)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (50257, 768)


def test_load_safetensors_mmap(safetensors_path):
    d = load_safetensors_mmap(safetensors_path)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (50257, 768)


# --- Backend: async ---

@pytest.mark.asyncio
async def test_open_async(safetensors_path):
    handle = await open_safetensors(safetensors_path)
    assert "wte" in handle.keys() and "wpe" in handle.keys()
    assert handle.get_tensor("wte").shape == (50257, 768)


@pytest.mark.asyncio
async def test_load_safetensors_async(safetensors_path):
    d = await load_safetensors(safetensors_path)
    assert "wte" in d and "wpe" in d
    assert d["wte"].shape == (50257, 768)


# --- Backend parity (sync/mmap/async consistency) ---

def test_load_keys_parity(safetensors_path):
    keys_sync = set(load_safetensors_sync(safetensors_path).keys())
    keys_mmap = set(load_safetensors_mmap(safetensors_path).keys())
    assert keys_sync == keys_mmap


@pytest.mark.asyncio
async def test_load_keys_parity_async(safetensors_path):
    keys_sync = set(load_safetensors_sync(safetensors_path).keys())
    keys_async = set((await load_safetensors(safetensors_path)).keys())
    assert keys_sync == keys_async


def test_load_shapes_parity(safetensors_path):
    d_sync = load_safetensors_sync(safetensors_path)
    d_mmap = load_safetensors_mmap(safetensors_path)
    assert {k: tuple(d_sync[k].shape) for k in d_sync} == {k: tuple(d_mmap[k].shape) for k in d_mmap}


@pytest.mark.asyncio
async def test_load_shapes_parity_async(safetensors_path):
    d_sync = load_safetensors_sync(safetensors_path)
    d_async = await load_safetensors(safetensors_path)
    assert {k: tuple(d_sync[k].shape) for k in d_sync} == {k: tuple(d_async[k].shape) for k in d_async}


def test_open_handle_parity(safetensors_path):
    h_sync = open_safetensors_sync(safetensors_path)
    h_mmap = open_safetensors_mmap(safetensors_path)
    assert set(h_sync.keys()) == set(h_mmap.keys())
    for k in h_sync.keys():
        assert h_sync.get_tensor(k).shape == h_mmap.get_tensor(k).shape
        assert torch.equal(h_sync.get_tensor(k), h_mmap.get_tensor(k))


@pytest.mark.asyncio
async def test_open_handle_parity_async(safetensors_path):
    h_sync = open_safetensors_sync(safetensors_path)
    h_async = await open_safetensors(safetensors_path)
    assert set(h_sync.keys()) == set(h_async.keys())
    for k in h_sync.keys():
        assert h_sync.get_tensor(k).shape == h_async.get_tensor(k).shape
        assert torch.equal(h_sync.get_tensor(k), h_async.get_tensor(k))


# --- Async concurrency ---

@pytest.mark.asyncio
@pytest.mark.slow
async def test_parallel_open_safetensors(safetensors_path):
    handles = await asyncio.gather(
        open_safetensors(safetensors_path),
        open_safetensors(safetensors_path),
        open_safetensors(safetensors_path),
    )
    for h in handles:
        assert "wte" in h.keys()
        assert h.get_tensor("wte").shape == (50257, 768)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_parallel_load_safetensors(safetensors_path):
    results = await asyncio.gather(
        load_safetensors(safetensors_path),
        load_safetensors(safetensors_path),
    )
    d1, d2 = results
    assert set(d1.keys()) == set(d2.keys())
    for k in d1:
        assert torch.equal(d1[k], d2[k])


@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_get_tensor_on_shared_handle(safetensors_path):
    handle = await open_safetensors(safetensors_path)
    keys = list(handle.keys())[:5]
    tensors = await asyncio.gather(
        *[asyncio.to_thread(handle.get_tensor, k) for k in keys]
    )
    for k, t in zip(keys, tensors):
        assert t.shape == handle.get_tensor(k).shape
        assert torch.equal(t, handle.get_tensor(k))
