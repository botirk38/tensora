#!/bin/bash
set -euo pipefail

cd ~/PROJECT/bindings/python
RESULTS_DIR=~/PROJECT/results/h100/vllm
mkdir -p "$RESULTS_DIR/raw"
SUMMARY="$RESULTS_DIR/qwen3_8b_io_uring_case.tsv"
echo -e "loader\tkind\tcache_mode\tmetric\tvalue" >"$SUMMARY"

MODEL_ID="Qwen/Qwen3-8B"
LOADER="ts_safetensors_io_uring"
KINDS=(load_only ttft steady_state_decode)

for cache_mode in warm cold; do
  for kind in "${KINDS[@]}"; do
    echo "=== $cache_mode $LOADER $kind ==="
    if [ "$cache_mode" = "cold" ]; then
      sync
      echo 3 >/proc/sys/vm/drop_caches
    fi
    output=$(.venv/bin/python -m benchmarks.vllm_runner --loader "$LOADER" --benchmark-kind "$kind" --model-id "$MODEL_ID" 2>&1)
    raw_path="$RESULTS_DIR/raw/${LOADER}_${kind}_${cache_mode}.log"
    printf '%s\n' "$output" >"$raw_path"
    python3 - "$raw_path" "$SUMMARY" "$LOADER" "$kind" "$cache_mode" <<'PY'
import json, sys
raw_path, summary, loader, kind, cache_mode = sys.argv[1:]
with open(raw_path) as f:
    lines = [line.strip() for line in f if line.strip()]
for line in reversed(lines):
    if line.startswith('{'):
        data = json.loads(line)
        break
else:
    raise SystemExit(f"No JSON payload found in {raw_path}")
with open(summary, 'a') as f:
    for key, value in data.items():
        f.write(f"{loader}\t{kind}\t{cache_mode}\t{key}\t{value}\n")
PY
  done
done
