"""SafeTensors benchmarks: native file baselines vs tensora directory loads."""

from safetensors.torch import load_file as safetensors_load_file

from benchmarks.hub_model import touch_tensor
from tensora._tensora_rust import (
    load_safetensors,
    load_safetensors_async,
    load_safetensors_sync,
    open_safetensors,
)


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


def test_load_native(benchmark, safetensors_files):
    """Benchmark native safetensors.torch.load_file across all shards."""

    def run():
        return _load_all_files(safetensors_files, safetensors_load_file)

    result = benchmark(run)
    assert result > 0


def test_load_tensorstore_sync(benchmark, safetensors_dir):
    """Benchmark tensora load_safetensors_sync for the full model directory."""

    def run():
        return _load_dir(safetensors_dir, load_safetensors_sync)

    result = benchmark(run)
    assert result > 0


def test_load_async(benchmark, safetensors_dir):
    """Benchmark tensora load_safetensors_async for the full model directory."""

    def run():
        return _load_dir(safetensors_dir, load_safetensors_async)

    result = benchmark(run)
    assert result > 0


def test_load_default(benchmark, safetensors_dir):
    """Benchmark tensora load_safetensors default backend for the full model directory."""

    def run():
        return _load_dir(safetensors_dir, load_safetensors)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor(benchmark, safetensors_dir):
    """Benchmark open_safetensors + get_tensor for the full model directory."""

    def run():
        return _open_dir(safetensors_dir, open_safetensors)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_default(benchmark, safetensors_dir):
    """Benchmark open_safetensors (default) + get_tensor for the full model directory."""

    def run():
        return _open_dir(safetensors_dir, open_safetensors)

    result = benchmark(run)
    assert result > 0
