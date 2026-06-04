"""Domain entity: Report — collects results, formats TSV, and analyzes separability."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from tensora_experiments.result import CellResult


@dataclass(slots=True)
class Report:
    """Aggregates CellResult instances and provides output/analysis methods.

    Attributes:
        name: Experiment name (used in file naming and headers).
        results: Collected profiling results.
    """

    name: str
    results: list[CellResult] = field(default_factory=list)

    @property
    def successes(self) -> list[CellResult]:
        return [r for r in self.results if r.succeeded]

    @property
    def failures(self) -> list[CellResult]:
        return [r for r in self.results if r.failed]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.successes) / len(self.results)

    def add(self, result: CellResult) -> None:
        """Append a single result."""
        self.results.append(result)

    def extend(self, results: list[CellResult]) -> None:
        """Append multiple results."""
        self.results.extend(results)

    def to_matrix_tsv(self) -> str:
        """Format as TSV matching results/h100/rust.tsv schema.

        Columns: model, format, backend, time_ms, tensors, bytes
        """
        header = "model\tformat\tbackend\ttime_ms\ttensors\tbytes"
        rows = [header] + [r.to_matrix_row() for r in self.results]
        return "\n".join(rows) + "\n"

    def to_anchor_tsv(self) -> str:
        """Format as TSV matching results/h100/anchor.tsv schema.

        Columns: model, format, backend, rep, time_ms, tensors, bytes
        """
        header = "model\tformat\tbackend\trep\ttime_ms\ttensors\tbytes"
        rows = [header] + [r.to_anchor_row() for r in self.results]
        return "\n".join(rows) + "\n"

    def write_tsv(self, output_dir: Path, anchor: bool = False) -> Path:
        """Write results to a TSV file in the given directory.

        Args:
            output_dir: Directory to write into (created if missing).
            anchor: If True, use anchor schema (with rep column).

        Returns:
            Path to the written file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.name.replace(' ', '_')}.tsv"
        filepath = output_dir / filename

        content = self.to_anchor_tsv() if anchor else self.to_matrix_tsv()
        filepath.write_text(content)
        return filepath

    def summary(self) -> str:
        """Human-readable summary of the report."""
        lines = [
            f"Report: {self.name}",
            f"  Total results: {len(self.results)}",
            f"  Successes:     {len(self.successes)}",
            f"  Failures:      {len(self.failures)}",
            f"  Success rate:  {self.success_rate:.1%}",
        ]
        if self.failures:
            lines.append("  Failed cells:")
            for f in self.failures:
                lines.append(f"    {f.model} / {f.format} / {f.backend} rep={f.rep}: {f.error}")
        return "\n".join(lines)

    def preview(self, anchor: bool = False, max_rows: int = 12) -> str:
        """Return the first N rows of the TSV output for display."""
        tsv = self.to_anchor_tsv() if anchor else self.to_matrix_tsv()
        lines = tsv.split("\n")
        preview_lines = ["--- TSV Preview ---"]
        for line in lines[:max_rows]:
            preview_lines.append(f"  {line}")
        remaining = len(self.results) - (max_rows - 1)
        if remaining > 0:
            preview_lines.append(f"  ... ({remaining} more rows)")
        return "\n".join(preview_lines)

    def separability_analysis(self) -> str:
        """Analyze whether ranges overlap for close-call backend pairs.

        Groups results by (model, format) and compares backend medians/ranges
        to determine if the ranking is statistically separable.
        """
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for r in self.successes:
            grouped[(r.model, r.format, r.backend)].append(r.time_ms)

        # Group by (model, format) to compare backends
        model_format_groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
        for (model, fmt, backend), times in grouped.items():
            model_format_groups[(model, fmt)][backend] = times

        lines: list[str] = ["Separability Analysis", "=" * 40]

        for (model, fmt), backends in sorted(model_format_groups.items()):
            lines.append(f"\n{model} / {fmt}:")

            # Compute stats per backend
            stats: list[tuple[str, float, float, float]] = []
            for backend, times in sorted(backends.items()):
                med = median(times)
                stats.append((backend, min(times), med, max(times)))
                lines.append(
                    f"  {backend:>10}: median={med:.2f}ms "
                    f"range=[{min(times):.2f}, {max(times):.2f}] "
                    f"n={len(times)}"
                )

            # Pairwise overlap check for adjacent-ranked backends
            stats.sort(key=lambda s: s[2])  # sort by median
            for i in range(len(stats) - 1):
                name_a, min_a, _, max_a = stats[i]
                name_b, min_b, _, max_b = stats[i + 1]
                overlaps = min_a <= max_b and min_b <= max_a
                if overlaps:
                    lines.append(f"  → {name_a} vs {name_b}: ranges OVERLAP")
                else:
                    lines.append(f"  → {name_a} vs {name_b}: SEPARABLE")

        return "\n".join(lines)
