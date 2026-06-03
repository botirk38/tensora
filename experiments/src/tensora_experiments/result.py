"""Domain entity: CellResult — one cold-cache profiling measurement."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ITERATION_PATTERN = re.compile(r"iteration (\d+): (\d+) tensors, (\d+) bytes, ([0-9.]+)ms")


@dataclass(frozen=True, slots=True)
class CellResult:
    """Immutable record of a single profiling iteration.

    Attributes:
        model: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B").
        format: Checkpoint format ("safetensors" or "serverlessllm").
        backend: I/O backend ("sync", "async", "io-uring", "default").
        rep: Repetition number (1-indexed).
        time_ms: Wall-clock time in milliseconds.
        tensors: Number of tensors loaded.
        bytes_loaded: Total bytes read.
        error: Error message if the cell failed, None on success.
    """

    model: str
    format: str
    backend: str
    rep: int
    time_ms: float
    tensors: int
    bytes_loaded: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_matrix_row(self) -> str:
        """Serialize as a single TSV row (matrix schema: no rep column)."""
        if self.error:
            return f"{self.model}\t{self.format}\t{self.backend}\tERROR\t0\t0\t{self.error}"
        return (
            f"{self.model}\t{self.format}\t{self.backend}\t"
            f"{self.time_ms:.2f}\t{self.tensors}\t{self.bytes_loaded}"
        )

    def to_anchor_row(self) -> str:
        """Serialize as a single TSV row (anchor schema: includes rep column)."""
        if self.error:
            return (
                f"{self.model}\t{self.format}\t{self.backend}\t{self.rep}\t"
                f"ERROR\t0\t0\t{self.error}"
            )
        return (
            f"{self.model}\t{self.format}\t{self.backend}\t{self.rep}\t"
            f"{self.time_ms:.2f}\t{self.tensors}\t{self.bytes_loaded}"
        )

    @classmethod
    def from_profile_output(
        cls,
        stdout: str,
        model: str,
        fmt: str,
        backend: str,
        rep_offset: int = 0,
    ) -> list[CellResult]:
        """Parse the tensora profile binary stdout into CellResult instances.

        Each 'iteration N: ...' line becomes one CellResult.
        rep_offset shifts the rep numbering for multi-call anchoring.
        """
        results: list[CellResult] = []
        for match in _ITERATION_PATTERN.finditer(stdout):
            iter_num = int(match.group(1))
            tensors = int(match.group(2))
            bytes_loaded = int(match.group(3))
            time_ms = float(match.group(4))
            results.append(
                cls(
                    model=model,
                    format=fmt,
                    backend=backend,
                    rep=rep_offset + iter_num,
                    time_ms=time_ms,
                    tensors=tensors,
                    bytes_loaded=bytes_loaded,
                )
            )
        return results

    @classmethod
    def error_result(
        cls,
        model: str,
        fmt: str,
        backend: str,
        rep: int,
        error: str,
    ) -> CellResult:
        """Construct a failed CellResult."""
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
