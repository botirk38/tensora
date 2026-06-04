"""Domain entity: Profiler — Modal cls that executes profiling cells on H100.

Uses class-based lifecycle (@modal.enter) to verify capabilities once per
container, then handles multiple profiling calls via @modal.method.
"""

from __future__ import annotations

import subprocess

import modal

from tensora_experiments.infrastructure import (
    EPHEMERAL_DISK_MIB,
    GPU,
    HF_CACHE_MOUNT,
    MEMORY_MIB,
    PROFILE_BIN,
    TIMEOUT_S,
    app,
    hf_volume,
    image,
)
from tensora_experiments.result import CellResult


@app.cls(
    image=image,
    gpu=GPU,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_S,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={HF_CACHE_MOUNT: hf_volume},
)
class Profiler:
    """Stateful profiling executor on Modal H100 hardware.

    Lifecycle:
        - @modal.enter: validates that the profile binary exists and reports
          backend capabilities (io_uring availability, etc.)
        - run_cell: executes a single profiling cell and returns parsed results.
    """

    @modal.enter()
    def validate_environment(self) -> None:
        """Run once per container: verify binary and log capabilities."""
        result = subprocess.run(
            [PROFILE_BIN, "capabilities"],
            capture_output=True,
            text=True,
            check=False,
        )
        self._capabilities = result.stdout.strip()
        print(f"[tensora] Profile binary: {PROFILE_BIN}")
        print(f"[tensora] Capabilities:\n{self._capabilities}")

    @modal.method()
    def run_cell(
        self,
        model_id: str,
        fmt: str,
        backend: str,
        iterations: int = 1,
        evict: bool = True,
        rep_offset: int = 0,
    ) -> list[CellResult]:
        """Execute one profiling cell.

        Args:
            model_id: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B").
            fmt: Checkpoint format ("safetensors" or "serverlessllm").
            backend: I/O backend ("sync", "async", "io-uring", "default").
            iterations: Number of iterations in a single binary invocation.
            evict: Whether to evict page cache between iterations.
            rep_offset: Base offset for rep numbering.

        Returns:
            List of CellResult, one per iteration.
        """
        cmd = [
            PROFILE_BIN,
            fmt,
            backend,
            "--model-id",
            model_id,
            "--iterations",
            str(iterations),
        ]
        if evict:
            cmd.append("--evict-page-cache")

        print(f"[tensora] Executing: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            check=False,
        )

        print(result.stdout)
        if result.stderr:
            print(f"[tensora:stderr] {result.stderr}")

        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"exit code {result.returncode}"
            return [
                CellResult.error_result(
                    model=model_id,
                    fmt=fmt,
                    backend=backend,
                    rep=rep_offset + 1,
                    error=error_msg,
                )
            ]

        results = CellResult.from_profile_output(result.stdout, model_id, fmt, backend, rep_offset)

        if not results:
            return [
                CellResult.error_result(
                    model=model_id,
                    fmt=fmt,
                    backend=backend,
                    rep=rep_offset + 1,
                    error=f"no iterations parsed from output: {result.stdout[:200]}",
                )
            ]

        return results

    @modal.method()
    def capabilities(self) -> str:
        """Return the backend capabilities string for this container."""
        return self._capabilities
