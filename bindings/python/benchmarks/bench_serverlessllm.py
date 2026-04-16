"""ServerlessLLM benchmarks for real models.

Benchmarks tensora ServerlessLLM loading with different eager backends plus
the lazy open/get_tensor path. Partition count uses the shared size-based heuristic.
"""


from benchmarks.hub_model import touch_tensor
from tensora._tensora_rust import (
    load_serverlessllm,
    load_serverlessllm_async,
    load_serverlessllm_sync,
    open_serverlessllm,
)


def test_load_async(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm_async."""

    def run():
        result = load_serverlessllm_async(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_sync(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm_sync."""

    def run():
        result = load_serverlessllm_sync(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_default(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm (default backend)."""

    def run():
        result = load_serverlessllm(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_open_get_tensor(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm + get_tensor."""

    def run():
        handle = open_serverlessllm(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_default(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm (default) + get_tensor."""

    def run():
        handle = open_serverlessllm(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)
