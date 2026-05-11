"""vLLM integration benchmarks: compare native vs tensora loaders.

Each loader runs load_only, ttft, and steady_state_decode in warm and cold cache modes.
Uses explicit loaders via vLLM's registered model loader mechanism.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.vllm_runner import BENCHMARK_KINDS, LOADERS

CACHE_MODES = ["warm", "cold"]

_GPU_FREE_TIMEOUT = 30


def _wait_for_gpu_idle() -> None:
    """Poll nvidia-smi until GPU memory is <5% used, up to timeout."""
    deadline = time.monotonic() + _GPU_FREE_TIMEOUT
    while time.monotonic() < deadline:
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                used, total = (int(x) for x in r.stdout.strip().split(","))
                if used < total * 0.05:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(2)


def _drop_linux_page_cache() -> None:
    """Best-effort host page-cache drop for cold-cache runs (Linux; usually requires root)."""
    drop = Path("/proc/sys/vm/drop_caches")
    if not drop.is_file():
        return
    try:
        subprocess.run(["sync"], check=False, timeout=300)
        drop.write_text("3", encoding="utf-8")
    except OSError:
        pass


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("benchmark_kind", BENCHMARK_KINDS)
@pytest.mark.parametrize("cache_mode", CACHE_MODES)
def test_vllm(benchmark, model_id, loader, benchmark_kind, cache_mode):
    """Parametrized vLLM benchmark: loader x benchmark_kind x cache_mode."""

    def run_subprocess():
        if cache_mode == "cold":
            _drop_linux_page_cache()
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
            timeout=1200,
        )

        _wait_for_gpu_idle()

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

    # One subprocess load per test: default benchmark(...) repeats many rounds and would
    # relaunch vLLM repeatedly (OOM / spurious failures / huge runtime).
    data = benchmark.pedantic(run_subprocess, rounds=1, iterations=1)
    benchmark.extra_info.update(data)

    if benchmark_kind == "load_only":
        assert "init_ms" in data
    elif benchmark_kind == "ttft":
        assert "ttft_ms" in data
    elif benchmark_kind == "steady_state_decode":
        assert "decode_avg_ms" in data
