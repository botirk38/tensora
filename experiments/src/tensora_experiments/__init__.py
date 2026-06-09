"""Tensora experiment harness — Modal-based cold-cache profiling on H100.

This library provides domain entities for defining, executing, and reporting
on Tensora profiling experiments via Modal's serverless GPU infrastructure.
"""

from tensora_experiments.config import CellSpec, ExperimentMatrix
from tensora_experiments.infrastructure import app
from tensora_experiments.profiler import Profiler
from tensora_experiments.report import Report
from tensora_experiments.result import CellResult
from tensora_experiments.vllm_config import VllmCellSpec, VllmExperimentMatrix
from tensora_experiments.vllm_profiler import VllmProfiler
from tensora_experiments.vllm_report import VllmReport
from tensora_experiments.vllm_result import VllmCellResult

__all__ = [
    "CellResult",
    "CellSpec",
    "ExperimentMatrix",
    "Profiler",
    "Report",
    "VllmCellResult",
    "VllmCellSpec",
    "VllmExperimentMatrix",
    "VllmProfiler",
    "VllmReport",
    "app",
]
