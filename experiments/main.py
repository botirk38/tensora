"""CLI entrypoint for running Tensora experiments on Modal.

Usage:
    modal run experiments/main.py --experiment held-out-validation
    modal run experiments/main.py --experiment full-anchor-matrix
    modal run experiments/main.py --experiment targeted-anchors
    modal run experiments/main.py --experiment held-out-validation --reps 5
"""

from pathlib import Path

from tensora_experiments import ExperimentMatrix, Profiler, Report, app


@app.local_entrypoint()
def run(
    experiment: str = "held-out-validation",
    reps: int = 0,
    output_dir: str = "results/modal",
    retry_tsv: str = "",
) -> None:
    """Execute a named experiment on Modal H100 infrastructure.

    Args:
        experiment: One of "held-out-validation", "full-anchor-matrix", "targeted-anchors".
        reps: Override repetition count (0 = use experiment default).
        output_dir: Directory for TSV output files.
        retry_tsv: Path to a TSV file with ERROR rows to retry. Overrides --experiment.
    """
    if retry_tsv:
        matrix = ExperimentMatrix.retry_errors(retry_tsv)
    else:
        matrix = ExperimentMatrix.from_name(experiment, reps_override=reps)

    print(matrix.summary())
    print()

    profiler = Profiler()

    print("--- Verifying H100 backend capabilities ---")
    print(profiler.capabilities.remote())
    print()

    print(f"--- Executing {matrix.total_runs} runs ---")
    report = Report(name=matrix.name)

    for result_batch in profiler.run_cell.starmap(matrix.to_starmap_args()):
        report.extend(result_batch)

    print()
    print(report.summary())

    out_path = report.write_tsv(Path(output_dir), anchor=matrix.anchor_mode)
    print(f"\nResults written to: {out_path}")

    if matrix.anchor_mode:
        print()
        print(report.separability_analysis())

    print()
    print(report.preview(anchor=matrix.anchor_mode))
