"""Generic Report that collects any BenchmarkResult and writes TSV via csv module."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

from tensora_experiments.result import BenchmarkResult


@dataclass(slots=True)
class Report[R: BenchmarkResult]:
    """Collects benchmark results and writes well-formed TSV output.

    Works identically for RustResult and VllmResult because both
    implement the BenchmarkResult protocol (tsv_columns / to_tsv_dict).
    """

    name: str
    results: list[R] = field(default_factory=list)

    # ── accessors ─────────────────────────────────────────────────

    @property
    def successes(self) -> list[R]:
        return [r for r in self.results if r.succeeded]

    @property
    def failures(self) -> list[R]:
        return [r for r in self.results if not r.succeeded]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.successes) / len(self.results)

    # ── mutation ──────────────────────────────────────────────────

    def add(self, result: R) -> None:
        self.results.append(result)

    def extend(self, results: list[R]) -> None:
        self.results.extend(results)

    # ── TSV serialisation (via csv.DictWriter) ────────────────────

    def to_tsv(self) -> str:
        """Render all results as a TSV string."""
        if not self.results:
            return ""

        columns = type(self.results[0]).tsv_columns()
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for r in self.results:
            writer.writerow(r.to_tsv_dict())
        return buf.getvalue()

    def write_tsv(self, output_dir: Path) -> Path:
        """Write results TSV to *output_dir/<name>.tsv*."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{self.name.replace(' ', '_')}.tsv"
        filepath.write_text(self.to_tsv())
        return filepath

    # ── human-readable output ─────────────────────────────────────

    def summary(self) -> str:
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
                row = f.to_tsv_dict()
                lines.append(f"    {row}")
        return "\n".join(lines)

    def preview(self, max_rows: int = 12) -> str:
        tsv = self.to_tsv()
        tsv_lines = tsv.split("\n")
        preview_lines = [f"--- TSV Preview ({self.name}) ---"]
        for line in tsv_lines[: max_rows + 1]:
            preview_lines.append(f"  {line}")
        remaining = len(self.results) - max_rows
        if remaining > 0:
            preview_lines.append(f"  ... ({remaining} more rows)")
        return "\n".join(preview_lines)
