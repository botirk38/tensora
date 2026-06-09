"""Domain entity: VllmCellResult — one vLLM benchmark measurement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VllmCellResult:
    """Immutable record of a single vLLM benchmark run.

    Attributes:
        model: HuggingFace model ID.
        loader: Loader name (e.g. "native", "ts_safetensors_sync").
        benchmark_kind: "load_only" or "ttft".
        rep: Repetition number (1-indexed).
        init_ms: vLLM initialization time in milliseconds.
        ttft_ms: Time to first token (init + generation) in milliseconds.
        first_token_ms: Generation-only time for first token.
        error: Error message if the cell failed, None on success.
    """

    model: str
    loader: str
    benchmark_kind: str
    rep: int
    init_ms: float
    ttft_ms: float = 0.0
    first_token_ms: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_tsv_row(self) -> str:
        """Serialize as a single TSV row."""
        if self.error:
            return (
                f"{self.model}\t{self.loader}\t{self.benchmark_kind}\t{self.rep}\t"
                f"ERROR\t0\t0\t{self.error}"
            )
        return (
            f"{self.model}\t{self.loader}\t{self.benchmark_kind}\t{self.rep}\t"
            f"{self.init_ms:.2f}\t{self.ttft_ms:.2f}\t{self.first_token_ms:.2f}"
        )

    @classmethod
    def error_result(
        cls,
        model: str,
        loader: str,
        benchmark_kind: str,
        rep: int,
        error: str,
    ) -> VllmCellResult:
        """Construct a failed VllmCellResult."""
        return cls(
            model=model,
            loader=loader,
            benchmark_kind=benchmark_kind,
            rep=rep,
            init_ms=0.0,
            ttft_ms=0.0,
            first_token_ms=0.0,
            error=error,
        )
