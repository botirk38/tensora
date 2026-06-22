"""SafeTensors benchmarks: native file baselines vs tensora backends + access patterns."""

import platform
import random

from safetensors.torch import load_file as safetensors_load_file

from benchmarks.hub_model import touch_tensor
from tensora._tensora_rust import (
    load_safetensors,
    open_safetensors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_all_files(files, loader_fn):
    total_count = 0
    for path in files:
        result = loader_fn(path)
        total_count += sum(touch_tensor(t) for t in result.values())
    return total_count


def _load_dir(path, loader_fn):
    result = loader_fn(path)
    return sum(touch_tensor(t) for t in result.values())


def _open_dir(path, open_fn):
    handle = open_fn(path)
    total_count = 0
    for k in handle.keys():
        total_count += touch_tensor(handle.get_tensor(k))
    return total_count


# ---------------------------------------------------------------------------
# Native safetensors baseline
# ---------------------------------------------------------------------------


def test_load_native(benchmark, safetensors_files, model_descriptor):
    """Native safetensors.torch.load_file across all shards."""
    benchmark.extra_info["total_bytes"] = model_descriptor.total_bytes
    benchmark.extra_info["shard_count"] = model_descriptor.shard_count

    def run():
        _load_all_files(safetensors_files, safetensors_load_file)

    benchmark(run)


# ---------------------------------------------------------------------------
# Full-model load benchmarks (one per backend)
# ---------------------------------------------------------------------------


def test_load_default(benchmark, safetensors_dir, model_descriptor):
    """tensora load_safetensors with the default backend."""
    benchmark.extra_info["total_bytes"] = model_descriptor.total_bytes

    def run():
        _load_dir(safetensors_dir, load_safetensors)

    benchmark(run)


def test_load_sync(benchmark, safetensors_dir, model_descriptor):
    """tensora load_safetensors sync backend."""
    benchmark.extra_info["total_bytes"] = model_descriptor.total_bytes

    def run():
        _load_dir(safetensors_dir, lambda p: load_safetensors(p, backend="sync"))

    benchmark(run)


def test_load_async(benchmark, safetensors_dir, model_descriptor):
    """tensora load_safetensors async backend."""
    benchmark.extra_info["total_bytes"] = model_descriptor.total_bytes

    def run():
        _load_dir(safetensors_dir, lambda p: load_safetensors(p, backend="async"))

    benchmark(run)


def test_load_io_uring(benchmark, safetensors_dir, model_descriptor):
    """tensora load_safetensors io_uring backend (Linux only)."""
    if platform.system() != "Linux":
        benchmark.extra_info["skipped"] = "io_uring requires Linux"
        return
    benchmark.extra_info["total_bytes"] = model_descriptor.total_bytes

    def run():
        _load_dir(safetensors_dir, lambda p: load_safetensors(p, backend="io_uring"))

    benchmark(run)


# ---------------------------------------------------------------------------
# Mmap-based lazy load + page touch
# ---------------------------------------------------------------------------


def test_open_get_tensor(benchmark, safetensors_dir, model_descriptor):
    """open_safetensors + sequential get_tensor (mmap, touches all pages)."""
    benchmark.extra_info["total_bytes"] = model_descriptor.total_bytes

    def run():
        _open_dir(safetensors_dir, open_safetensors)

    benchmark(run)


# ---------------------------------------------------------------------------
# Tensor access pattern benchmarks (pre-loaded model)
# ---------------------------------------------------------------------------


def test_tensor_sequential(benchmark, safetensors_dir, model_descriptor):
    """Sequential tensor access on pre-loaded model."""
    result = load_safetensors(safetensors_dir, backend="sync")
    keys = list(result.keys())
    benchmark.extra_info["tensor_count"] = len(keys)

    def run():
        total = 0
        for k in keys:
            total += touch_tensor(result[k])
        return total

    benchmark(run)


def test_tensor_random(benchmark, safetensors_dir, model_descriptor):
    """Random-order tensor access on pre-loaded model."""
    result = load_safetensors(safetensors_dir, backend="sync")
    keys = list(result.keys())
    shuffled = keys.copy()
    random.seed(42)
    random.shuffle(shuffled)
    benchmark.extra_info["tensor_count"] = len(keys)

    def run():
        total = 0
        for k in shuffled:
            total += touch_tensor(result[k])
        return total

    benchmark(run)
