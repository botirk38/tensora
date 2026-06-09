"""vLLM Profiler — Modal class for vLLM integration benchmarks on H100.

Measures how I/O backend choice affects vLLM initialization (load_only),
time-to-first-token (TTFT), and steady-state decode throughput.
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

from tensora_experiments.enums import BenchmarkKind, VllmLoader
from tensora_experiments.infrastructure import (
    BUILD,
    COMPUTE,
    HF_CACHE_MOUNT,
    app,
    hf_volume,
    vllm_image,
)
from tensora_experiments.result import VllmResult

_VLLM_ENV: dict[str, str] = {
    "HF_HOME": HF_CACHE_MOUNT,
    "HF_HUB_DISABLE_XET": "1",
    "VLLM_LOGGING_LEVEL": "CRITICAL",
    "VLLM_USE_DEEP_GEMM": "0",
    "VLLM_MOE_USE_DEEP_GEMM": "0",
    "VLLM_DEEP_GEMM_WARMUP": "skip",
    "TOKENIZERS_PARALLELISM": "false",
}


@app.cls(
    image=vllm_image,
    gpu=COMPUTE.gpu,
    ephemeral_disk=COMPUTE.ephemeral_disk_mib,
    memory=COMPUTE.memory_mib,
    timeout=COMPUTE.timeout_s,
    retries=modal.Retries(
        max_retries=COMPUTE.max_retries,
        backoff_coefficient=COMPUTE.backoff_coefficient,
        initial_delay=COMPUTE.initial_delay_s,
    ),
    volumes={HF_CACHE_MOUNT: hf_volume},
)
class VllmProfiler:
    """Stateful vLLM benchmark executor on Modal H100 hardware."""

    _env_info: str

    @modal.enter()
    def validate_environment(self) -> None:
        python_bin = f"{BUILD.workspace}/bindings/python/.venv/bin/python"
        result = subprocess.run(
            [
                python_bin,
                "-c",
                "import tensora; import vllm; print(f'tensora OK, vLLM {vllm.__version__}')",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self._env_info = result.stdout.strip()
        print(f"[vllm-profiler] Environment: {self._env_info}")
        if result.returncode != 0:
            print(f"[vllm-profiler:stderr] {result.stderr}")

    @modal.method()
    def run_cell(
        self,
        model_id: str,
        loader: str,
        benchmark_kind: str,
        rep_offset: int = 0,
    ) -> list[VllmResult]:
        """Execute one vLLM benchmark cell."""
        loader_enum = VllmLoader(loader)
        kind_enum = BenchmarkKind(benchmark_kind)

        python_bin = f"{BUILD.workspace}/bindings/python/.venv/bin/python"
        cmd = [
            python_bin,
            "-m",
            "benchmarks.vllm_runner",
            "--loader",
            loader,
            "--benchmark-kind",
            benchmark_kind,
            "--model-id",
            model_id,
        ]

        print(f"[vllm-profiler] Executing: {' '.join(cmd)}")

        env = {
            **os.environ,
            "PATH": ("/root/.cargo/bin:" + os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")),
            **_VLLM_ENV,
        }

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=COMPUTE.subprocess_timeout_s,
            check=False,
            cwd=f"{BUILD.workspace}/bindings/python",
            env=env,
        )

        print(
            result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        )
        if result.stderr:
            print(f"[vllm-profiler:stderr] {result.stderr[-1000:]}")

        if result.returncode != 0:
            error_msg = result.stderr.strip()[-500:] or f"exit code {result.returncode}"
            return [
                VllmResult.error_result(
                    model=model_id,
                    loader=loader_enum,
                    benchmark_kind=kind_enum,
                    rep=rep_offset + 1,
                    error=error_msg,
                )
            ]

        parsed = _parse_json_output(result.stdout)
        if parsed is None:
            return [
                VllmResult.error_result(
                    model=model_id,
                    loader=loader_enum,
                    benchmark_kind=kind_enum,
                    rep=rep_offset + 1,
                    error=(f"no JSON parsed from output: {result.stdout[:200]}"),
                )
            ]

        return [
            VllmResult(
                model=model_id,
                loader=loader_enum,
                benchmark_kind=kind_enum,
                rep=rep_offset + 1,
                init_ms=parsed.get("init_ms", 0.0),
                ttft_ms=parsed.get("ttft_ms", 0.0),
                first_token_ms=parsed.get("first_token_ms", 0.0),
                decode_avg_ms=parsed.get("decode_avg_ms", 0.0),
                decode_min_ms=parsed.get("decode_min_ms", 0.0),
                decode_max_ms=parsed.get("decode_max_ms", 0.0),
            )
        ]

    @modal.method()
    def capabilities(self) -> str:
        return self._env_info


def _parse_json_output(stdout: str) -> dict | None:
    """Extract the last JSON object from process stdout."""
    for line in reversed(stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
