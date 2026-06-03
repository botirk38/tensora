"""Domain entity: ExperimentMatrix — defines the profiling space to explore."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class CellSpec:
    """A single cell in the experiment matrix.

    Represents one (model, format, backend) combination with a repetition count.
    """

    model: str
    format: str
    backend: str
    reps: int = 1

    def to_starmap_args(self) -> list[tuple[str, str, str, int, bool, int]]:
        """Expand into Modal .starmap() argument tuples — one per rep.

        Returns:
            List of (model_id, fmt, backend, iterations=1, evict=True, rep_offset).
        """
        return [(self.model, self.format, self.backend, 1, True, r) for r in range(self.reps)]


@dataclass(slots=True)
class ExperimentMatrix:
    """Defines the complete experiment space to profile.

    Supports two construction modes:
    - Cartesian product: from models x formats x backends x reps
    - Explicit cells: from a list of CellSpec instances

    Provides factory methods for the professor's suggestions.
    """

    name: str
    models: list[str] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)
    reps: int = 1
    explicit_cells: list[CellSpec] = field(default_factory=list)

    @property
    def total_cells(self) -> int:
        if self.explicit_cells:
            return len(self.explicit_cells)
        return len(self.models) * len(self.formats) * len(self.backends)

    @property
    def total_runs(self) -> int:
        if self.explicit_cells:
            return sum(c.reps for c in self.explicit_cells)
        return self.total_cells * self.reps

    def cells(self) -> list[CellSpec]:
        """Generate all CellSpec instances."""
        if self.explicit_cells:
            return self.explicit_cells
        return [
            CellSpec(model=m, format=f, backend=b, reps=self.reps)
            for m, f, b in product(self.models, self.formats, self.backends)
        ]

    def to_starmap_args(self) -> list[tuple[str, str, str, int, bool, int]]:
        """Flatten all cells into Modal .starmap() argument tuples."""
        args: list[tuple[str, str, str, int, bool, int]] = []
        for cell in self.cells():
            args.extend(cell.to_starmap_args())
        return args

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [f"Experiment: {self.name}"]
        if self.explicit_cells:
            lines.append(f"  Mode:     explicit ({len(self.explicit_cells)} cells)")
            for cell in self.explicit_cells:
                lines.append(f"    - {cell.model} / {cell.format} / {cell.backend} x{cell.reps}")
        else:
            lines.extend(
                [
                    f"  Models:   {self.models}",
                    f"  Formats:  {self.formats}",
                    f"  Backends: {self.backends}",
                    f"  Reps:     {self.reps}",
                ]
            )
        lines.append(f"  Total:    {self.total_runs} runs across {self.total_cells} cells")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Registry: resolve experiment name to configuration
    # ------------------------------------------------------------------

    REGISTRY: ClassVar[dict[str, str]] = {
        "held-out-validation": "held_out_validation",
        "full-anchor-matrix": "full_anchor_matrix",
        "targeted-anchors": "targeted_anchors",
    }

    @classmethod
    def from_name(cls, name: str, reps_override: int = 0) -> ExperimentMatrix:
        """Resolve an experiment name to an ExperimentMatrix.

        Args:
            name: Experiment identifier (e.g. "held-out-validation").
            reps_override: If >0, override the default rep count.

        Raises:
            ValueError: If name is not in the registry.
        """
        if name not in cls.REGISTRY:
            valid = list(cls.REGISTRY.keys())
            msg = f"Unknown experiment: {name!r}. Valid: {valid}"
            raise ValueError(msg)

        method_name = cls.REGISTRY[name]
        factory = getattr(cls, method_name)

        if name == "targeted-anchors":
            return factory()

        reps = reps_override if reps_override > 0 else (1 if name == "held-out-validation" else 5)
        return factory(reps=reps)

    # ------------------------------------------------------------------
    # Factory methods for predefined experiment configurations
    # ------------------------------------------------------------------

    @classmethod
    def held_out_validation(cls, reps: int = 1) -> ExperimentMatrix:
        """Issue #22 / Suggestion B: held-out model validation.

        Profiles Llama-3.1-8B and Mistral-7B across the full backend/format
        matrix to test whether the adaptive selector generalizes beyond
        Qwen-family checkpoint geometry.
        """
        return cls(
            name="held-out-validation",
            models=[
                "meta-llama/Llama-3.1-8B",
                "mistralai/Mistral-7B-Instruct-v0.3",
            ],
            formats=["safetensors", "serverlessllm"],
            backends=["sync", "async", "io-uring", "default"],
            reps=reps,
        )

    @classmethod
    def full_anchor_matrix(cls, reps: int = 5) -> ExperimentMatrix:
        """Issue #23 / Suggestion C: full 5-rep matrix for statistical anchoring.

        All 6 original paper models x 2 formats x 4 backends x 5 reps = 240 runs.
        """
        return cls(
            name="full-anchor-matrix",
            models=[
                "Qwen/Qwen3-0.6B",
                "HuggingFaceTB/SmolLM3-3B",
                "Qwen/Qwen3-4B",
                "Qwen/Qwen3-8B",
                "Qwen/Qwen3-14B",
                "Qwen/Qwen3-32B",
            ],
            formats=["safetensors", "serverlessllm"],
            backends=["sync", "async", "io-uring", "default"],
            reps=reps,
        )

    @classmethod
    def targeted_anchors(cls) -> ExperimentMatrix:
        """Issue #23 / Suggestion C: the 3 close-call cells with specific backends.

        These cells have <3% gap between top-two backends and need anchoring
        to confirm or refute the observed ranking.
        """
        return cls(
            name="targeted-anchors",
            explicit_cells=[
                CellSpec(model="Qwen/Qwen3-14B", format="safetensors", backend="io-uring", reps=5),
                CellSpec(model="Qwen/Qwen3-14B", format="safetensors", backend="default", reps=5),
                CellSpec(
                    model="Qwen/Qwen3-14B", format="serverlessllm", backend="io-uring", reps=5
                ),
                CellSpec(model="Qwen/Qwen3-14B", format="serverlessllm", backend="default", reps=5),
                CellSpec(model="Qwen/Qwen3-4B", format="serverlessllm", backend="async", reps=5),
                CellSpec(model="Qwen/Qwen3-4B", format="serverlessllm", backend="default", reps=5),
            ],
        )
