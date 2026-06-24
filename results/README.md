# Results

Archived experiment data and benchmark outputs.

## Structure

- `modal/` — Primary 5-rep experiments on Modal H100 (gVisor container runtime)
  - `full-anchor-matrix.tsv` — 6 models × 2 formats × 3 backends × 5 reps (180 runs)
  - `hf-native-baseline.tsv` — 6 models × SafeTensors × native HF loader × 5 reps (30 runs)
  - `held-out-validation.tsv` — 2 held-out models × 2 formats × 3 backends × 5 reps (60 runs)
  - `vllm-load-only.tsv` — 5 models × 5 loaders × 5 reps (125 runs)
- `h100/` — Bare-metal H100 measurements (1 rep) + predicted 5-rep variants
- `h200/` — Bare-metal H200 measurements (1 rep) + predicted 5-rep variants
- `a100/` — A100 measurements (1 rep) + predicted 5-rep variants

## Predicted 5-rep files (`*_5rep_predicted.tsv`)

Single-rep measurements from H100/H200/A100 are expanded to 5 synthetic reps
using the coefficient of variation (CV) observed in Modal's actual 5-rep runs.
For each (model, format, backend) cell, the CV is taken from the corresponding
Modal cell (warm reps 2–5 only, excluding cold-start rep 1). When no exact
match exists, a per-(format, backend) median CV is used as fallback.

Samples are drawn from N(observed, observed × CV) with seed 42 for reproducibility.

## Notes

- Published numbers use cold-cache methodology from the paper
- Replicate **ranking and regime behaviour**, not byte-identical timings
- TSV/JSON in `h100/` are reference baselines, not normative
