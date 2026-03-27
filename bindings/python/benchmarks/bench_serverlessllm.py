"""ServerlessLLM benchmarks: load_file and open+get_tensor per backend (default, sync, mmap).
Also benchmarks parallel loading with multiple partitions.
"""

import pytest

from benchmarks.fixtures import drop_page_cache, touch_tensor
from tensor_store_py._tensor_store_rust import (
    load_serverlessllm,
    load_serverlessllm_sync,
    open_serverlessllm,
    open_serverlessllm_mmap,
    open_serverlessllm_sync,
)


def test_load_sync_warm(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm_sync (warm cache)."""

    def run():
        result = load_serverlessllm_sync(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_sync_cold(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm_sync (cold cache, drop before each run)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        result = load_serverlessllm_sync(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_default_warm(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm (default backend, warm cache)."""

    def run():
        result = load_serverlessllm(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_default_cold(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm (default backend, cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        result = load_serverlessllm(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_open_get_tensor_sync_warm(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm_sync + get_tensor (warm, touch all pages)."""

    def run():
        handle = open_serverlessllm_sync(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_sync_cold(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm_sync + get_tensor (cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        handle = open_serverlessllm_sync(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_mmap_warm(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm_mmap + get_tensor (warm, touch all pages)."""

    def run():
        handle = open_serverlessllm_mmap(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_mmap_cold(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm_mmap + get_tensor (cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        handle = open_serverlessllm_mmap(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_default_warm(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm (default) + get_tensor (warm, touch all pages)."""

    def run():
        handle = open_serverlessllm(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_default_cold(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm (default) + get_tensor (cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        handle = open_serverlessllm(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


# =============================================================================
# Multi-partition parallel loading benchmarks
# =============================================================================


def test_load_sync_4partitions_warm(benchmark, serverlessllm_dir_4partitions):
    """Benchmark load with 4 partitions (parallel loading)."""

    def run():
        result = load_serverlessllm_sync(serverlessllm_dir_4partitions)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_sync_8partitions_warm(benchmark, serverlessllm_dir_8partitions):
    """Benchmark load with 8 partitions (parallel loading)."""

    def run():
        result = load_serverlessllm_sync(serverlessllm_dir_8partitions)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_sync_4partitions_cold(benchmark, serverlessllm_dir_4partitions):
    """Benchmark load with 4 partitions (cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir_4partitions)
        result = load_serverlessllm_sync(serverlessllm_dir_4partitions)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_sync_8partitions_cold(benchmark, serverlessllm_dir_8partitions):
    """Benchmark load with 8 partitions (cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir_8partitions)
        result = load_serverlessllm_sync(serverlessllm_dir_8partitions)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0
