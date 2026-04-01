"""vLLM integration benchmarks: compare native vs tensor_store loaders.

Each loader runs load_only, ttft, and steady_state_decode in warm and cold cache modes.
Uses explicit loaders via vLLM's registered model loader mechanism.
"""

import json
import subprocess
import sys

import pytest


LOADERS = [
    "native",
    "ts_safetensors_default",
    "ts_safetensors_sync",
    "ts_safetensors_io_uring",
    "ts_serverlessllm_default",
    "ts_serverlessllm_sync",
    "ts_serverlessllm_io_uring",
]

BENCHMARK_KINDS = ["load_only", "ttft", "steady_state_decode"]
CACHE_MODES = ["warm", "cold"]


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("benchmark_kind", BENCHMARK_KINDS)
@pytest.mark.parametrize("cache_mode", CACHE_MODES)
def test_vllm(benchmark, model_id, loader, benchmark_kind, cache_mode):
    """Parametrized vLLM benchmark: loader x benchmark_kind x cache_mode."""

    def run_subprocess():
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.vllm_runner",
                "--loader",
                loader,
                "--benchmark-kind",
                benchmark_kind,
                "--model-id",
                model_id,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(f"vLLM subprocess failed: {result.stderr}")

        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        raise RuntimeError("Could not parse JSON from vLLM runner")

    data = benchmark(run_subprocess)

    if benchmark_kind == "load_only":
        assert "init_ms" in data
    elif benchmark_kind == "ttft":
        assert "ttft_ms" in data
    elif benchmark_kind == "steady_state_decode":
        assert "decode_avg_ms" in data
