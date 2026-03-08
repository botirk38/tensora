"""SafeTensors benchmarks: load_file and open+get_tensor per backend (sync, mmap, async)."""

import asyncio

import pytest

from benchmarks.fixtures import drop_page_cache, touch_tensor
from tensor_store_py import (
    load_safetensors_sync,
    open_safetensors,
    open_safetensors_mmap,
    open_safetensors_sync,
)


def test_load_sync_warm(benchmark, safetensors_path):
    """Benchmark load_safetensors_sync (warm cache)."""
    def run():
        result = load_safetensors_sync(safetensors_path)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_sync_cold(benchmark, safetensors_path):
    """Benchmark load_safetensors_sync (cold cache, drop before each run)."""
    def run():
        drop_page_cache(safetensors_path)
        result = load_safetensors_sync(safetensors_path)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_open_get_tensor_sync_warm(benchmark, safetensors_path):
    """Benchmark open_safetensors_sync + get_tensor (warm, touch all pages)."""

    def run():
        handle = open_safetensors_sync(safetensors_path)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_sync_cold(benchmark, safetensors_path):
    """Benchmark open_safetensors_sync + get_tensor (cold cache)."""

    def run():
        drop_page_cache(safetensors_path)
        handle = open_safetensors_sync(safetensors_path)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_mmap_warm(benchmark, safetensors_path):
    """Benchmark open_safetensors_mmap + get_tensor (warm, touch all pages)."""

    def run():
        handle = open_safetensors_mmap(safetensors_path)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_mmap_cold(benchmark, safetensors_path):
    """Benchmark open_safetensors_mmap + get_tensor (cold cache)."""

    def run():
        drop_page_cache(safetensors_path)
        handle = open_safetensors_mmap(safetensors_path)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_async_warm(benchmark, safetensors_path):
    """Benchmark open_safetensors (async) + get_tensor (warm, touch all pages)."""

    async def run():
        handle = await open_safetensors(safetensors_path)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    def run_sync():
        asyncio.run(run())

    benchmark(run_sync)


def test_open_get_tensor_async_cold(benchmark, safetensors_path):
    """Benchmark open_safetensors (async) + get_tensor (cold cache)."""

    async def run():
        handle = await open_safetensors(safetensors_path)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    def run_sync():
        drop_page_cache(safetensors_path)
        asyncio.run(run())

    benchmark(run_sync)
