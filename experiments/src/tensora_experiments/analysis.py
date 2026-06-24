"""Statistical analysis utilities for Rust profiling results."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from statistics import median

from tensora_experiments.result import RustResult


def separability_analysis(results: list[RustResult]) -> str:
    """Check whether backend rankings are statistically separable.

    Groups successful results by (model, format), then for each group
    compares backend medians and ranges to determine if consecutive
    backends in the ranking have non-overlapping ranges.
    """
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in results:
        if r.succeeded:
            grouped[(r.model, r.format.value, r.backend.value)].append(
                r.time_ms,
            )

    model_format_groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
    for (model, fmt, backend), times in grouped.items():
        model_format_groups[(model, fmt)][backend] = times

    lines: list[str] = ["Separability Analysis", "=" * 40]

    for (model, fmt), backends in sorted(model_format_groups.items()):
        lines.append(f"\n{model} / {fmt}:")

        stats: list[tuple[str, float, float, float]] = []
        for backend, times in sorted(backends.items()):
            med = median(times)
            stats.append((backend, min(times), med, max(times)))
            lines.append(
                f"  {backend:>10}: median={med:.2f}ms "
                f"range=[{min(times):.2f}, {max(times):.2f}] "
                f"n={len(times)}"
            )

        stats.sort(key=lambda s: s[2])
        for (name_a, min_a, _, max_a), (name_b, min_b, _, max_b) in pairwise(
            stats,
        ):
            overlaps = min_a <= max_b and min_b <= max_a
            verdict = "ranges OVERLAP" if overlaps else "SEPARABLE"
            lines.append(f"  -> {name_a} vs {name_b}: {verdict}")

    return "\n".join(lines)
