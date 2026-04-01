# Scripts

This directory now keeps only the scripts that were actually useful in the final experiment flow.

## Keepers

- `download_models.py`
  - downloads SafeTensors fixtures from Hugging Face
  - optionally converts them to `ServerlessLLM`
- `run_anchor_reps.sh`
  - runs 5 cold reps on the Rust anchor set
  - writes `results/h100/profile/anchor_reps.tsv`
- `run_full_cold_matrix.sh`
  - runs the full Rust cold matrix across formats, fixtures, and backends
  - writes `results/h100/profile/full_cold_matrix.tsv`
- `run_vllm_full_matrix.sh`
  - runs the full vLLM matrix across the fixture ladder and all benchmark loaders
  - writes `results/h100/vllm/full_matrix.tsv`
- `rerun_failed_vllm.sh`
  - reruns only the failed rows from the vLLM matrix after harness/config changes

## Setup

```bash
cd scripts
uv sync
```

## Typical Usage

Download fixtures:

```bash
uv run python download_models.py Qwen/Qwen3-8B --verify
```

Run Rust anchor reps:

```bash
./scripts/run_anchor_reps.sh
```

Run the full Rust cold matrix:

```bash
./scripts/run_full_cold_matrix.sh
```

Run the full vLLM matrix:

```bash
./scripts/run_vllm_full_matrix.sh
```

Rerun only failed vLLM rows after a harness update:

```bash
./scripts/rerun_failed_vllm.sh
```

## Notes

- Results are saved under `results/h100/...`.
- The one-off debugging runners used during development were removed on purpose.
- `run_vllm_full_matrix.sh` is resumable: it skips rows that are already recorded.
- `rerun_failed_vllm.sh` rewrites failed rows in place after a successful retry.
