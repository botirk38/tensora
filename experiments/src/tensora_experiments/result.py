"""Benchmark result types with a shared Protocol for generic reporting.

Both RustResult and VllmResult conform to BenchmarkResult so they can
be collected by the unified Report[R] class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tensora_experiments.enums import (
    Backend,
    BenchmarkKind,
    Format,
    VllmLoader,
)

# ── Protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class BenchmarkResult(Protocol):
    """Minimal interface every result type must satisfy."""

    @property
    def succeeded(self) -> bool: ...

    @classmethod
    def tsv_columns(cls) -> list[str]: ...

    def to_tsv_dict(self) -> dict[str, str]: ...


# ── Rust profiling result ─────────────────────────────────────────────

_ITERATION_RE = re.compile(
    r"iteration (\d+): (\d+) tensors, (\d+) bytes, ([0-9.]+)ms",
)


@dataclass(frozen=True, slots=True)
class RustResult:
    """One cold-cache profiling iteration from the Rust binary."""

    model: str
    format: Format
    backend: Backend
    rep: int
    time_ms: float
    tensors: int
    bytes_loaded: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @classmethod
    def tsv_columns(cls) -> list[str]:
        return [
            "model",
            "format",
            "backend",
            "rep",
            "time_ms",
            "tensors",
            "bytes",
        ]

    def to_tsv_dict(self) -> dict[str, str]:
        if self.error:
            return {
                "model": self.model,
                "format": self.format.value,
                "backend": self.backend.value,
                "rep": str(self.rep),
                "time_ms": "ERROR",
                "tensors": "0",
                "bytes": "0",
                "error": _escape_tsv(self.error),
            }
        return {
            "model": self.model,
            "format": self.format.value,
            "backend": self.backend.value,
            "rep": str(self.rep),
            "time_ms": f"{self.time_ms:.2f}",
            "tensors": str(self.tensors),
            "bytes": str(self.bytes_loaded),
        }

    @classmethod
    def from_profile_output(
        cls,
        stdout: str,
        model: str,
        fmt: Format,
        backend: Backend,
        rep_offset: int = 0,
    ) -> list[RustResult]:
        """Parse the tensora profile binary stdout into results."""
        results: list[RustResult] = []
        for match in _ITERATION_RE.finditer(stdout):
            results.append(
                cls(
                    model=model,
                    format=fmt,
                    backend=backend,
                    rep=rep_offset + int(match.group(1)),
                    time_ms=float(match.group(4)),
                    tensors=int(match.group(2)),
                    bytes_loaded=int(match.group(3)),
                )
            )
        return results

    @classmethod
    def error_result(
        cls,
        model: str,
        fmt: Format,
        backend: Backend,
        rep: int,
        error: str,
    ) -> RustResult:
        return cls(
            model=model,
            format=fmt,
            backend=backend,
            rep=rep,
            time_ms=0.0,
            tensors=0,
            bytes_loaded=0,
            error=error,
        )


# ── vLLM benchmark result ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VllmResult:
    """One vLLM benchmark run (load_only, TTFT, or steady-state decode)."""

    model: str
    loader: VllmLoader
    benchmark_kind: BenchmarkKind
    rep: int
    init_ms: float
    ttft_ms: float = 0.0
    first_token_ms: float = 0.0
    decode_avg_ms: float = 0.0
    decode_min_ms: float = 0.0
    decode_max_ms: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @classmethod
    def tsv_columns(cls) -> list[str]:
        return [
            "model",
            "loader",
            "benchmark_kind",
            "rep",
            "init_ms",
            "ttft_ms",
            "first_token_ms",
            "decode_avg_ms",
            "decode_min_ms",
            "decode_max_ms",
        ]

    def to_tsv_dict(self) -> dict[str, str]:
        if self.error:
            return {
                "model": self.model,
                "loader": self.loader.value,
                "benchmark_kind": self.benchmark_kind.value,
                "rep": str(self.rep),
                "init_ms": "ERROR",
                "ttft_ms": "0",
                "first_token_ms": "0",
                "decode_avg_ms": "0",
                "decode_min_ms": "0",
                "decode_max_ms": "0",
                "error": _escape_tsv(self.error),
            }
        return {
            "model": self.model,
            "loader": self.loader.value,
            "benchmark_kind": self.benchmark_kind.value,
            "rep": str(self.rep),
            "init_ms": f"{self.init_ms:.2f}",
            "ttft_ms": f"{self.ttft_ms:.2f}",
            "first_token_ms": f"{self.first_token_ms:.2f}",
            "decode_avg_ms": f"{self.decode_avg_ms:.2f}",
            "decode_min_ms": f"{self.decode_min_ms:.2f}",
            "decode_max_ms": f"{self.decode_max_ms:.2f}",
        }

    @classmethod
    def error_result(
        cls,
        model: str,
        loader: VllmLoader,
        benchmark_kind: BenchmarkKind,
        rep: int,
        error: str,
    ) -> VllmResult:
        return cls(
            model=model,
            loader=loader,
            benchmark_kind=benchmark_kind,
            rep=rep,
            init_ms=0.0,
            error=error,
        )


# ── helpers ───────────────────────────────────────────────────────────


def _escape_tsv(text: str) -> str:
    """Escape tabs and newlines so error messages don't corrupt TSV."""
    return text.replace("\t", " ").replace("\n", " ").replace("\r", "")
