"""Rust Profiler — Modal class that executes cold-cache profiling on H100.

Uses @modal.enter for one-time environment validation and @modal.method
for the per-cell benchmark invocation.
"""

from __future__ import annotations

import subprocess

import modal

from tensora_experiments.enums import Backend, Format
from tensora_experiments.infrastructure import (
    BUILD,
    COMPUTE,
    HF_CACHE_MOUNT,
    app,
    hf_volume,
    rust_image,
)
from tensora_experiments.result import RustResult


@app.cls(
    image=rust_image,
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
class RustProfiler:
    """Stateful profiling executor on Modal H100 hardware."""

    _capabilities: str

    @modal.enter()
    def validate_environment(self) -> None:
        result = subprocess.run(
            [BUILD.profile_bin, "capabilities"],
            capture_output=True,
            text=True,
            check=False,
        )
        self._capabilities = result.stdout.strip()
        print(f"[tensora] Profile binary: {BUILD.profile_bin}")
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
    ) -> list[RustResult]:
        """Execute one profiling cell and return parsed results."""
        fmt_enum = Format(fmt)
        backend_enum = Backend(backend)

        cmd = [
            BUILD.profile_bin,
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
            timeout=COMPUTE.subprocess_timeout_s,
            check=False,
        )

        print(result.stdout)
        if result.stderr:
            print(f"[tensora:stderr] {result.stderr}")

        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"exit code {result.returncode}"
            return [
                RustResult.error_result(
                    model=model_id,
                    fmt=fmt_enum,
                    backend=backend_enum,
                    rep=rep_offset + 1,
                    error=error_msg,
                )
            ]

        results = RustResult.from_profile_output(
            result.stdout,
            model_id,
            fmt_enum,
            backend_enum,
            rep_offset,
        )

        if not results:
            return [
                RustResult.error_result(
                    model=model_id,
                    fmt=fmt_enum,
                    backend=backend_enum,
                    rep=rep_offset + 1,
                    error=(f"no iterations parsed from output: {result.stdout[:200]}"),
                )
            ]

        return results

    @modal.method()
    def capabilities(self) -> str:
        return self._capabilities
