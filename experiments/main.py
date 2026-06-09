"""CLI entrypoint for running Tensora experiments on Modal.

Usage:
    modal run experiments/main.py --experiment full-anchor-matrix
    modal run experiments/main.py --experiment vllm-full --reps 5
    modal run experiments/main.py --experiment hf-native-baseline --reps 3
"""

from pathlib import Path

from tensora_experiments.analysis import separability_analysis
from tensora_experiments.infrastructure import app
from tensora_experiments.matrix import RustMatrix, VllmMatrix
from tensora_experiments.profiler import RustProfiler
from tensora_experiments.report import Report
from tensora_experiments.result import RustResult, VllmResult
from tensora_experiments.vllm_profiler import VllmProfiler

_ALL_EXPERIMENTS = RustMatrix.EXPERIMENT_NAMES + VllmMatrix.EXPERIMENT_NAMES


@app.local_entrypoint()
def run(
    experiment: str = "full-anchor-matrix",
    reps: int = 0,
    output_dir: str = "results/modal",
) -> None:
    """Execute a named experiment on Modal H100 infrastructure.

    Args:
        experiment: Experiment name (Rust or vLLM).
        reps: Override repetition count (0 = use experiment default).
        output_dir: Directory for TSV output files.
    """
    if experiment in VllmMatrix.EXPERIMENT_NAMES:
        _run_vllm(experiment, reps, output_dir)
    elif experiment in RustMatrix.EXPERIMENT_NAMES:
        _run_rust(experiment, reps, output_dir)
    else:
        valid = ", ".join(_ALL_EXPERIMENTS)
        msg = f"Unknown experiment: {experiment!r}. Valid: {valid}"
        raise SystemExit(msg)


def _run_rust(experiment: str, reps: int, output_dir: str) -> None:
    effective_reps = reps if reps > 0 else 5
    matrix = RustMatrix.from_name(experiment, reps=effective_reps)

    print(matrix.summary())
    print()

    profiler = RustProfiler()

    print("--- Verifying H100 backend capabilities ---")
    print(profiler.capabilities.remote())
    print()

    print(f"--- Executing {matrix.total_runs} runs ---")
    report: Report[RustResult] = Report(name=matrix.name)

    for result_batch in profiler.run_cell.starmap(
        matrix.to_starmap_args(),
    ):
        report.extend(result_batch)

    print()
    print(report.summary())

    out_path = report.write_tsv(Path(output_dir))
    print(f"\nResults written to: {out_path}")

    if effective_reps > 1:
        print()
        print(separability_analysis(report.successes))

    print()
    print(report.preview())


def _run_vllm(experiment: str, reps: int, output_dir: str) -> None:
    effective_reps = reps if reps > 0 else 1
    matrix = VllmMatrix.from_name(experiment, reps=effective_reps)

    print(matrix.summary())
    print()

    profiler = VllmProfiler()

    print("--- Verifying vLLM environment ---")
    print(profiler.capabilities.remote())
    print()

    print(f"--- Executing {matrix.total_runs} vLLM runs ---")
    report: Report[VllmResult] = Report(name=matrix.name)

    for result_batch in profiler.run_cell.starmap(
        matrix.to_starmap_args(),
    ):
        report.extend(result_batch)

    print()
    print(report.summary())

    out_path = report.write_tsv(Path(output_dir))
    print(f"\nResults written to: {out_path}")

    print()
    print(report.preview())
