"""ServerlessLLM benchmarks: all backends, access patterns, mmap page-touch.

Benchmarks tensora ServerlessLLM loading with each backend variant, plus
the lazy open/get_tensor path. Partition count uses the shared size-based heuristic.
"""

import platform
import random

from benchmarks.hub_model import touch_tensor
from tensora._tensora_rust import (
    load_serverlessllm,
    open_serverlessllm,
)


# ---------------------------------------------------------------------------
# Full-model load benchmarks (one per backend)
# ---------------------------------------------------------------------------


def test_load_default(benchmark, serverlessllm_dir):
    """load_serverlessllm with default (adaptive) backend."""

    def run():
        result = load_serverlessllm(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())

    benchmark(run)


def test_load_sync(benchmark, serverlessllm_dir):
    """load_serverlessllm with sync backend."""

    def run():
        result = load_serverlessllm(serverlessllm_dir, backend="sync")
        sum(touch_tensor(t) for t in result.values())

    benchmark(run)


def test_load_async(benchmark, serverlessllm_dir):
    """load_serverlessllm with async backend."""

    def run():
        result = load_serverlessllm(serverlessllm_dir, backend="async")
        sum(touch_tensor(t) for t in result.values())

    benchmark(run)


def test_load_io_uring(benchmark, serverlessllm_dir):
    """load_serverlessllm with io_uring backend (Linux only)."""
    if platform.system() != "Linux":
        benchmark.extra_info["skipped"] = "io_uring requires Linux"
        return

    def run():
        result = load_serverlessllm(serverlessllm_dir, backend="io_uring")
        sum(touch_tensor(t) for t in result.values())

    benchmark(run)


# ---------------------------------------------------------------------------
# Mmap-based lazy load + page touch
# ---------------------------------------------------------------------------


def test_open_get_tensor(benchmark, serverlessllm_dir):
    """open_serverlessllm + sequential get_tensor (mmap)."""

    def run():
        handle = open_serverlessllm(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


# ---------------------------------------------------------------------------
# Tensor access pattern benchmarks (pre-loaded model)
# ---------------------------------------------------------------------------


def test_tensor_sequential(benchmark, serverlessllm_dir):
    """Sequential tensor access on pre-loaded model."""
    result = load_serverlessllm(serverlessllm_dir, backend="sync")
    keys = list(result.keys())

    def run():
        total = 0
        for k in keys:
            total += touch_tensor(result[k])
        return total

    benchmark(run)


def test_tensor_random(benchmark, serverlessllm_dir):
    """Random-order tensor access on pre-loaded model."""
    result = load_serverlessllm(serverlessllm_dir, backend="sync")
    keys = list(result.keys())
    shuffled = keys.copy()
    random.seed(42)
    random.shuffle(shuffled)

    def run():
        total = 0
        for k in shuffled:
            total += touch_tensor(result[k])
        return total

    benchmark(run)
