"""Domain entity: VllmProfiler — Modal cls for vLLM integration benchmarks on H100.

Measures how I/O backend choice affects vLLM initialization (load_only) and
time-to-first-token (TTFT) on Modal H100 hardware.
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

from tensora_experiments.infrastructure import (
    EPHEMERAL_DISK_MIB,
    GIT_COMMIT,
    GPU,
    HF_CACHE_MOUNT,
    MEMORY_MIB,
    REPO_BRANCH,
    REPO_URL,
    RUST_VERSION,
    TIMEOUT_S,
    WORKSPACE,
    app,
    hf_volume,
)
from tensora_experiments.vllm_result import VllmCellResult

# ---------------------------------------------------------------------------
# Container image: CUDA + Rust + tensora maturin build + vLLM
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "git",
        "curl",
        "ca-certificates",
    )
    .run_commands(
        f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        f"| sh -s -- -y --default-toolchain {RUST_VERSION}",
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HUB_DISABLE_XET": "1",
            "GIT_COMMIT": GIT_COMMIT,
        }
    )
    .run_commands(
        f"git clone --depth 1 --branch {REPO_BRANCH} {REPO_URL} {WORKSPACE}",
    )
    .pip_install("uv")
    .run_commands(
        f"cd {WORKSPACE}/bindings/python && "
        "uv sync --group torch && "
        "uv run maturin develop --release",
    )
    .run_commands(
        f"cd {WORKSPACE}/bindings/python && "
        "uv pip install 'vllm>=0.8,<1.0'",
    )
)

# ---------------------------------------------------------------------------
# vLLM loader configurations
# ---------------------------------------------------------------------------

VLLM_LOADERS: dict[str, dict | None] = {
    "native": None,
    "ts_safetensors_sync": {"format": "safetensors", "backend": "sync"},
    "ts_safetensors_async": {"format": "safetensors", "backend": "async"},
    "ts_serverlessllm_sync": {"format": "serverlessllm", "backend": "sync"},
    "ts_serverlessllm_async": {"format": "serverlessllm", "backend": "async"},
}


@app.cls(
    image=vllm_image,
    gpu=GPU,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_S,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={HF_CACHE_MOUNT: hf_volume},
)
class VllmProfiler:
    """Stateful vLLM benchmark executor on Modal H100 hardware.

    Runs vLLM initialization and inference benchmarks using the
    vllm_runner.py script from the tensora bindings.
    """

    @modal.enter()
    def validate_environment(self) -> None:
        """Verify tensora and vLLM are importable."""
        result = subprocess.run(
            [
                f"{WORKSPACE}/bindings/python/.venv/bin/python",
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
    ) -> list[VllmCellResult]:
        """Execute one vLLM benchmark cell.

        Args:
            model_id: HuggingFace model ID.
            loader: One of VLLM_LOADERS keys.
            benchmark_kind: "load_only" or "ttft".
            rep_offset: Base offset for rep numbering.

        Returns:
            List of VllmCellResult (one element for this single run).
        """
        python_bin = f"{WORKSPACE}/bindings/python/.venv/bin/python"
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

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
            cwd=f"{WORKSPACE}/bindings/python",
            env={
                **os.environ,
                "PATH": "/root/.cargo/bin:"
                + os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                "HF_HOME": HF_CACHE_MOUNT,
                "HF_HUB_DISABLE_XET": "1",
                "VLLM_LOGGING_LEVEL": "CRITICAL",
                "VLLM_USE_DEEP_GEMM": "0",
                "VLLM_MOE_USE_DEEP_GEMM": "0",
                "VLLM_DEEP_GEMM_WARMUP": "skip",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )

        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.stderr:
            print(f"[vllm-profiler:stderr] {result.stderr[-1000:]}")

        if result.returncode != 0:
            error_msg = result.stderr.strip()[-500:] or f"exit code {result.returncode}"
            return [
                VllmCellResult.error_result(
                    model=model_id,
                    loader=loader,
                    benchmark_kind=benchmark_kind,
                    rep=rep_offset + 1,
                    error=error_msg,
                )
            ]

        # Parse JSON output from vllm_runner
        parsed = None
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if parsed is None:
            return [
                VllmCellResult.error_result(
                    model=model_id,
                    loader=loader,
                    benchmark_kind=benchmark_kind,
                    rep=rep_offset + 1,
                    error=f"no JSON parsed from output: {result.stdout[:200]}",
                )
            ]

        return [
            VllmCellResult(
                model=model_id,
                loader=loader,
                benchmark_kind=benchmark_kind,
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
        """Return environment info string."""
        return self._env_info
