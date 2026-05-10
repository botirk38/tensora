"""vLLM integration benchmarks: compare native vs tensora loaders.

Each loader runs load_only, ttft, and steady_state_decode in warm and cold cache modes.
Uses explicit loaders via vLLM's registered model loader mechanism.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.vllm_runner import BENCHMARK_KINDS, LOADERS

CACHE_MODES = ["warm", "cold"]


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
        proc = subprocess.Popen(
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            raise
        finally:
            # Kill the entire process group to clean up EngineCore children
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            time.sleep(3)

        if proc.returncode != 0:
            raise RuntimeError(f"vLLM subprocess failed: {stderr}")

        for line in reversed(stdout.strip().split("\n")):
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

    if benchmark_kind == "load_only":
        assert "init_ms" in data
    elif benchmark_kind == "ttft":
        assert "ttft_ms" in data
    elif benchmark_kind == "steady_state_decode":
        assert "decode_avg_ms" in data
