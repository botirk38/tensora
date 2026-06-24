"""Tensora experiment harness — Modal-based cold-cache profiling on H100.

Public API re-exports for the experiment domain.
"""

from tensora_experiments.analysis import separability_analysis
from tensora_experiments.enums import Backend, BenchmarkKind, Format, VllmLoader
from tensora_experiments.infrastructure import app
from tensora_experiments.matrix import RustMatrix, VllmMatrix
from tensora_experiments.profiler import RustProfiler
from tensora_experiments.report import Report
from tensora_experiments.result import BenchmarkResult, RustResult, VllmResult
from tensora_experiments.vllm_profiler import VllmProfiler

__all__ = [
    "Backend",
    "BenchmarkKind",
    "BenchmarkResult",
    "Format",
    "Report",
    "RustMatrix",
    "RustProfiler",
    "RustResult",
    "VllmLoader",
    "VllmMatrix",
    "VllmProfiler",
    "VllmResult",
    "app",
    "separability_analysis",
]
