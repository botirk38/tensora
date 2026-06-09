"""Domain entity: VllmExperimentMatrix — defines the vLLM profiling space."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class VllmCellSpec:
    """A single cell in the vLLM experiment matrix.

    Represents one (model, loader, benchmark_kind) combination with a repetition count.
    """

    model: str
    loader: str
    benchmark_kind: str
    reps: int = 1

    def to_starmap_args(self) -> list[tuple[str, str, str, int]]:
        """Expand into Modal .starmap() argument tuples — one per rep.

        Returns:
            List of (model_id, loader, benchmark_kind, rep_offset).
        """
        return [(self.model, self.loader, self.benchmark_kind, r) for r in range(self.reps)]


@dataclass(slots=True)
class VllmExperimentMatrix:
    """Defines the complete vLLM experiment space.

    Supports Cartesian product: models x loaders x benchmark_kinds x reps.
    """

    name: str
    models: list[str] = field(default_factory=list)
    loaders: list[str] = field(default_factory=list)
    benchmark_kinds: list[str] = field(default_factory=list)
    reps: int = 1

    @property
    def total_cells(self) -> int:
        return len(self.models) * len(self.loaders) * len(self.benchmark_kinds)

    @property
    def total_runs(self) -> int:
        return self.total_cells * self.reps

    def cells(self) -> list[VllmCellSpec]:
        return [
            VllmCellSpec(model=m, loader=ldr, benchmark_kind=bk, reps=self.reps)
            for m, ldr, bk in product(self.models, self.loaders, self.benchmark_kinds)
        ]

    def to_starmap_args(self) -> list[tuple[str, str, str, int]]:
        args: list[tuple[str, str, str, int]] = []
        for cell in self.cells():
            args.extend(cell.to_starmap_args())
        return args

    def summary(self) -> str:
        lines = [
            f"vLLM Experiment: {self.name}",
            f"  Models:          {self.models}",
            f"  Loaders:         {self.loaders}",
            f"  Benchmark kinds: {self.benchmark_kinds}",
            f"  Reps:            {self.reps}",
            f"  Total:           {self.total_runs} runs across {self.total_cells} cells",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    EXPERIMENT_NAMES: ClassVar[list[str]] = [
        "vllm-load-only",
        "vllm-ttft",
        "vllm-full",
    ]

    @classmethod
    def from_name(cls, name: str, reps_override: int = 0) -> VllmExperimentMatrix:
        registry = {
            "vllm-load-only": cls.vllm_load_only,
            "vllm-ttft": cls.vllm_ttft,
            "vllm-full": cls.vllm_full,
        }
        if name not in registry:
            msg = f"Unknown vLLM experiment: {name!r}. Valid: {cls.EXPERIMENT_NAMES}"
            raise ValueError(msg)
        reps = reps_override if reps_override > 0 else 1
        return registry[name](reps=reps)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    # Modal-available loaders: native + tensora sync/async for both formats.
    # io_uring is unavailable on Modal (gVisor).
    _MODAL_LOADERS: ClassVar[list[str]] = [
        "native",
        "ts_safetensors_sync",
        "ts_safetensors_async",
        "ts_serverlessllm_sync",
        "ts_serverlessllm_async",
    ]

    # Representative model subset for vLLM (full ladder is too expensive).
    _VLLM_MODELS: ClassVar[list[str]] = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
    ]

    @classmethod
    def vllm_load_only(cls, reps: int = 1) -> VllmExperimentMatrix:
        """vLLM cold load_only: how backend choice affects initialization time."""
        return cls(
            name="vllm-load-only",
            models=cls._VLLM_MODELS,
            loaders=cls._MODAL_LOADERS,
            benchmark_kinds=["load_only"],
            reps=reps,
        )

    @classmethod
    def vllm_ttft(cls, reps: int = 1) -> VllmExperimentMatrix:
        """vLLM TTFT: how backend choice affects time-to-first-token."""
        return cls(
            name="vllm-ttft",
            models=cls._VLLM_MODELS,
            loaders=cls._MODAL_LOADERS,
            benchmark_kinds=["ttft"],
            reps=reps,
        )

    @classmethod
    def vllm_full(cls, reps: int = 1) -> VllmExperimentMatrix:
        """vLLM full: both load_only and TTFT."""
        return cls(
            name="vllm-full",
            models=cls._VLLM_MODELS,
            loaders=cls._MODAL_LOADERS,
            benchmark_kinds=["load_only", "ttft"],
            reps=reps,
        )
