"""Experiment matrix definitions for Rust and vLLM benchmarks.

Each matrix is a Cartesian product of its dimensions (models, formats,
backends, etc.) expanded to (cell_spec, rep) pairs for Modal starmap.
Starmap arguments are typed tuples, not bare positional tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import ClassVar, NamedTuple

from tensora_experiments.enums import (
    Backend,
    BenchmarkKind,
    Format,
    VllmLoader,
)

# ── Named starmap argument types ──────────────────────────────────────


class RustStarmapArgs(NamedTuple):
    """Positional args for Profiler.run_cell via Modal starmap."""

    model_id: str
    fmt: str
    backend: str
    iterations: int
    evict: bool
    rep_offset: int


class VllmStarmapArgs(NamedTuple):
    """Positional args for VllmProfiler.run_cell via Modal starmap."""

    model_id: str
    loader: str
    benchmark_kind: str
    rep_offset: int


# ── Rust experiment matrix ────────────────────────────────────────────

# Shared model ladder (the six Qwen/SmolLM checkpoints used in the paper).
MODEL_LADDER: list[str] = [
    "Qwen/Qwen3-0.6B",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]

# Modal-available backends (gVisor blocks io_uring).
MODAL_BACKENDS: list[Backend] = [Backend.SYNC, Backend.ASYNC, Backend.DEFAULT]

# Modal-available vLLM loaders.
MODAL_LOADERS: list[VllmLoader] = list(VllmLoader)

# vLLM model subset (omits SmolLM3-3B to save GPU time).
VLLM_MODELS: list[str] = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]


@dataclass(frozen=True, slots=True)
class RustMatrix:
    """Cartesian product: models x formats x backends x reps."""

    name: str
    models: list[str]
    formats: list[Format]
    backends: list[Backend]
    reps: int = 1

    @property
    def total_cells(self) -> int:
        return len(self.models) * len(self.formats) * len(self.backends)

    @property
    def total_runs(self) -> int:
        return self.total_cells * self.reps

    def to_starmap_args(self) -> list[RustStarmapArgs]:
        args: list[RustStarmapArgs] = []
        for model, fmt, backend in product(self.models, self.formats, self.backends):
            for rep in range(self.reps):
                args.append(
                    RustStarmapArgs(
                        model_id=model,
                        fmt=fmt.value,
                        backend=backend.value,
                        iterations=1,
                        evict=True,
                        rep_offset=rep,
                    )
                )
        return args

    def summary(self) -> str:
        fmts = [f.value for f in self.formats]
        backs = [b.value for b in self.backends]
        return (
            f"Experiment: {self.name}\n"
            f"  Models:   {self.models}\n"
            f"  Formats:  {fmts}\n"
            f"  Backends: {backs}\n"
            f"  Reps:     {self.reps}\n"
            f"  Total:    {self.total_runs} runs across"
            f" {self.total_cells} cells"
        )

    # ── Registry ──────────────────────────────────────────────────

    EXPERIMENT_NAMES: ClassVar[list[str]] = [
        "full-anchor-matrix",
        "held-out-validation",
        "hf-native-baseline",
    ]

    _REGISTRY: ClassVar[
        dict[str, tuple[type[RustMatrix], dict]]  # type: ignore[type-arg]
    ] = {}

    @classmethod
    def _build_registry(cls) -> None:
        if cls._REGISTRY:
            return
        cls._REGISTRY = {
            "full-anchor-matrix": (
                cls,
                {
                    "name": "full-anchor-matrix",
                    "models": MODEL_LADDER,
                    "formats": [Format.SAFETENSORS, Format.SERVERLESSLLM],
                    "backends": MODAL_BACKENDS,
                },
            ),
            "held-out-validation": (
                cls,
                {
                    "name": "held-out-validation",
                    "models": [
                        "Qwen/Qwen3-8B",
                        "mistralai/Mistral-7B-Instruct-v0.3",
                    ],
                    "formats": [Format.SAFETENSORS, Format.SERVERLESSLLM],
                    "backends": MODAL_BACKENDS,
                },
            ),
            "hf-native-baseline": (
                cls,
                {
                    "name": "hf-native-baseline",
                    "models": MODEL_LADDER,
                    "formats": [Format.SAFETENSORS],
                    "backends": [Backend.HF_NATIVE],
                },
            ),
        }

    @classmethod
    def from_name(cls, name: str, reps: int = 5) -> RustMatrix:
        cls._build_registry()
        if name not in cls._REGISTRY:
            valid = ", ".join(cls.EXPERIMENT_NAMES)
            msg = f"Unknown experiment: {name!r}. Valid: {valid}"
            raise ValueError(msg)
        _, kwargs = cls._REGISTRY[name]
        return cls(**kwargs, reps=reps)


# ── vLLM experiment matrix ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VllmMatrix:
    """Cartesian product: models x loaders x benchmark_kinds x reps."""

    name: str
    models: list[str]
    loaders: list[VllmLoader]
    benchmark_kinds: list[BenchmarkKind]
    reps: int = 1

    @property
    def total_cells(self) -> int:
        return len(self.models) * len(self.loaders) * len(self.benchmark_kinds)

    @property
    def total_runs(self) -> int:
        return self.total_cells * self.reps

    def to_starmap_args(self) -> list[VllmStarmapArgs]:
        args: list[VllmStarmapArgs] = []
        for model, loader, kind in product(self.models, self.loaders, self.benchmark_kinds):
            for rep in range(self.reps):
                args.append(
                    VllmStarmapArgs(
                        model_id=model,
                        loader=loader.value,
                        benchmark_kind=kind.value,
                        rep_offset=rep,
                    )
                )
        return args

    def summary(self) -> str:
        ldrs = [ld.value for ld in self.loaders]
        kinds = [k.value for k in self.benchmark_kinds]
        return (
            f"vLLM Experiment: {self.name}\n"
            f"  Models:          {self.models}\n"
            f"  Loaders:         {ldrs}\n"
            f"  Benchmark kinds: {kinds}\n"
            f"  Reps:            {self.reps}\n"
            f"  Total:           {self.total_runs} runs across"
            f" {self.total_cells} cells"
        )

    # ── Registry ──────────────────────────────────────────────────

    EXPERIMENT_NAMES: ClassVar[list[str]] = [
        "vllm-load-only",
        "vllm-ttft",
        "vllm-decode",
        "vllm-full",
    ]

    @classmethod
    def from_name(cls, name: str, reps: int = 1) -> VllmMatrix:
        factories: dict[str, dict] = {
            "vllm-load-only": {
                "benchmark_kinds": [BenchmarkKind.LOAD_ONLY],
            },
            "vllm-ttft": {
                "benchmark_kinds": [BenchmarkKind.TTFT],
            },
            "vllm-decode": {
                "benchmark_kinds": [BenchmarkKind.STEADY_STATE_DECODE],
            },
            "vllm-full": {
                "benchmark_kinds": [
                    BenchmarkKind.LOAD_ONLY,
                    BenchmarkKind.TTFT,
                    BenchmarkKind.STEADY_STATE_DECODE,
                ],
            },
        }
        if name not in factories:
            valid = ", ".join(cls.EXPERIMENT_NAMES)
            msg = f"Unknown vLLM experiment: {name!r}. Valid: {valid}"
            raise ValueError(msg)
        return cls(
            name=name,
            models=VLLM_MODELS,
            loaders=MODAL_LOADERS,
            reps=reps,
            **factories[name],
        )
