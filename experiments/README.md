# Tensora Experiments

Modal-based experiment harness for running Tensora cold-cache profiling on H100 GPUs.

## Setup

```bash
cd experiments
uv sync
```

## Usage

Run experiments via Modal CLI from the **repo root**:

```bash
# Issue #22 / Suggestion B: held-out model validation (16 runs)
modal run experiments/main.py --experiment held-out-validation

# Issue #22 with 5 reps per cell (80 runs)
modal run experiments/main.py --experiment held-out-validation --reps 5

# Issue #23 / Suggestion C: full 5-rep matrix (240 runs)
modal run experiments/main.py --experiment full-anchor-matrix

# Issue #23 / Suggestion C: targeted close-call cells only (30 runs)
modal run experiments/main.py --experiment targeted-anchors
```

## Architecture

```
experiments/
├── main.py                          # CLI entrypoint (modal run)
├── pyproject.toml                   # uv-managed project config
└── src/
    └── tensora_experiments/         # Library package
        ├── __init__.py              # Public API exports
        ├── config.py                # ExperimentMatrix, CellSpec entities
        ├── infrastructure.py        # Modal app, image, volumes, constants
        ├── profiler.py              # Profiler Modal cls (lifecycle + execution)
        ├── report.py                # Report entity (collection, TSV, analysis)
        └── result.py                # CellResult entity (immutable measurement)
```

### Domain Entities

| Entity | Responsibility |
|--------|---------------|
| `CellResult` | Immutable record of one profiling iteration |
| `CellSpec` | One (model, format, backend, reps) specification |
| `ExperimentMatrix` | Defines the full experiment space; resolves names to configs |
| `Profiler` | Modal cls executing profiling cells on H100 with lifecycle |
| `Report` | Collects results, formats TSV, produces separability analysis |

### Modal Best Practices Applied

- `@app.cls` with `@modal.enter()` lifecycle hook for one-time environment validation
- `gpu="H100!"` disables auto-upgrade for benchmark reproducibility
- `modal.Volume` for persistent HuggingFace model cache across runs
- `ephemeral_disk=65536` for local NVMe (64 GiB)
- `.starmap()` for parallel fan-out across cells
- `modal.Retries` for transient failure recovery
- Proper image layering: system deps → Rust toolchain → binary build

## Output

Results are written as TSV files to `results/modal/`:

- Matrix schema: `model, format, backend, time_ms, tensors, bytes`
- Anchor schema: `model, format, backend, rep, time_ms, tensors, bytes`

## Development

```bash
cd experiments
uv run ruff check src/
uv run ruff format src/
uv run ty check src/
```
