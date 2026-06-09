"""Domain entity: VllmReport — collects vLLM results and formats TSV."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tensora_experiments.vllm_result import VllmCellResult


@dataclass(slots=True)
class VllmReport:
    """Aggregates VllmCellResult instances and provides output methods.

    Attributes:
        name: Experiment name (used in file naming and headers).
        results: Collected vLLM benchmark results.
    """

    name: str
    results: list[VllmCellResult] = field(default_factory=list)

    @property
    def successes(self) -> list[VllmCellResult]:
        return [r for r in self.results if r.succeeded]

    @property
    def failures(self) -> list[VllmCellResult]:
        return [r for r in self.results if r.failed]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.successes) / len(self.results)

    def add(self, result: VllmCellResult) -> None:
        self.results.append(result)

    def extend(self, results: list[VllmCellResult]) -> None:
        self.results.extend(results)

    def to_tsv(self) -> str:
        """Format as TSV.

        Columns: model, loader, benchmark_kind, rep, init_ms, ttft_ms, first_token_ms
        """
        header = "model\tloader\tbenchmark_kind\trep\tinit_ms\tttft_ms\tfirst_token_ms\tdecode_avg_ms\tdecode_min_ms\tdecode_max_ms"
        rows = [header] + [r.to_tsv_row() for r in self.results]
        return "\n".join(rows) + "\n"

    def write_tsv(self, output_dir: Path) -> Path:
        """Write results to a TSV file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.name.replace(' ', '_')}.tsv"
        filepath = output_dir / filename
        filepath.write_text(self.to_tsv())
        return filepath

    def summary(self) -> str:
        lines = [
            f"vLLM Report: {self.name}",
            f"  Total results: {len(self.results)}",
            f"  Successes:     {len(self.successes)}",
            f"  Failures:      {len(self.failures)}",
            f"  Success rate:  {self.success_rate:.1%}",
        ]
        if self.failures:
            lines.append("  Failed cells:")
            for failure in self.failures:
                lines.append(
                    f"    {failure.model} / {failure.loader} / "
                    f"{failure.benchmark_kind} rep={failure.rep}: {failure.error}"
                )
        return "\n".join(lines)

    def preview(self, max_rows: int = 12) -> str:
        tsv = self.to_tsv()
        lines = tsv.split("\n")
        preview_lines = ["--- vLLM TSV Preview ---"]
        for line in lines[:max_rows]:
            preview_lines.append(f"  {line}")
        remaining = len(self.results) - (max_rows - 1)
        if remaining > 0:
            preview_lines.append(f"  ... ({remaining} more rows)")
        return "\n".join(preview_lines)
