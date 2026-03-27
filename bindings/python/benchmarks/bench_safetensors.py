"""SafeTensors benchmarks: load_file and open+get_tensor per backend (default, sync, mmap).
Also compares native safetensors.torch.load_file vs tensor_store implementation.
"""

import pytest
from safetensors.torch import load_file as safetensors_load_file

from benchmarks.fixtures import drop_page_cache, touch_tensor
from tensor_store_py._tensor_store_rust import (
    load_safetensors,
    load_safetensors_mmap,
    load_safetensors_sync,
    open_safetensors,
    open_safetensors_mmap,
    open_safetensors_sync,
)


def test_load_native_warm(benchmark, safetensors_path):
    """Benchmark native safetensors.torch.load_file (warm cache)."""

    def run():
        result = safetensors_load_file(safetensors_path)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_native_cold(benchmark, safetensors_path):
    """Benchmark native safetensors.torch.load_file (cold cache)."""

    def run():
        drop_page_cache(safetensors_path)
        result = safetensors_load_file(safetensors_path)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_tensorstore_warm(benchmark, safetensors_path):
    """Benchmark tensor_store load_safetensors_sync (warm cache)."""

    def run():
        result = load_safetensors_sync(safetensors_path)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_tensorstore_cold(benchmark, safetensors_path):
    """Benchmark tensor_store load_safetensors_sync (cold cache)."""

    def run():
        drop_page_cache(safetensors_path)
        result = load_safetensors_sync(safetensors_path)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_mmap_warm(benchmark, safetensors_path):
    """Benchmark tensor_store load_safetensors_mmap (warm cache)."""

    def run():
        result = load_safetensors_mmap(safetensors_path)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_mmap_cold(benchmark, safetensors_path):
    """Benchmark tensor_store load_safetensors_mmap (cold cache)."""

    def run():
        drop_page_cache(safetensors_path)
        result = load_safetensors_mmap(safetensors_path)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_default_warm(benchmark, safetensors_path):
    """Benchmark tensor_store load_safetensors default backend (warm cache)."""

    def run():
        result = load_safetensors(safetensors_path)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_default_cold(benchmark, safetensors_path):
    """Benchmark tensor_store load_safetensors default backend (cold cache)."""

    def run():
        drop_page_cache(safetensors_path)
        result = load_safetensors(safetensors_path)
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


def test_open_get_tensor_default_warm(benchmark, safetensors_path):
    """Benchmark open_safetensors (default) + get_tensor (warm, touch all pages)."""

    def run():
        handle = open_safetensors(safetensors_path)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_default_cold(benchmark, safetensors_path):
    """Benchmark open_safetensors (default) + get_tensor (cold cache)."""

    def run():
        drop_page_cache(safetensors_path)
        handle = open_safetensors(safetensors_path)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)
