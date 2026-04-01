#!/bin/bash
set -euo pipefail

cd ~/PROJECT/bindings/python
RESULTS_DIR=~/PROJECT/results/h100/vllm
mkdir -p "$RESULTS_DIR/raw"
SUMMARY="$RESULTS_DIR/full_matrix.tsv"
if [ ! -f "$SUMMARY" ]; then
	printf 'model\tloader\tkind\tcache_mode\tmetric\tvalue\tstatus\n' >"$SUMMARY"
fi

MODELS=(
	'Qwen/Qwen3-0.6B'
	'HuggingFaceTB/SmolLM3-3B'
	'Qwen/Qwen3-4B'
	'Qwen/Qwen3-8B'
	'Qwen/Qwen3-14B'
	'Qwen/Qwen3-32B'
)
LOADERS=(
	native
	ts_safetensors_default
	ts_safetensors_sync
	ts_safetensors_io_uring
	ts_serverlessllm_default
	ts_serverlessllm_sync
	ts_serverlessllm_io_uring
)
KINDS=(load_only ttft steady_state_decode)

slugify() {
	echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's|/|-|g; s|[^a-z0-9._-]|-|g'
}

for model in "${MODELS[@]}"; do
	model_slug=$(slugify "$model")
	for cache_mode in warm cold; do
		for loader in "${LOADERS[@]}"; do
			for kind in "${KINDS[@]}"; do
				echo "=== $model $cache_mode $loader $kind ==="
				if grep -Fq "$model	$loader	$kind	$cache_mode	" "$SUMMARY"; then
					echo "  already recorded, skipping"
					continue
				fi
				if [ "$cache_mode" = "cold" ]; then
					sync
					echo 3 >/proc/sys/vm/drop_caches
				fi
				raw_path="$RESULTS_DIR/raw/${model_slug}_${loader}_${kind}_${cache_mode}.log"
				status=OK
				if ! output=$(.venv/bin/python -m benchmarks.vllm_runner --loader "$loader" --benchmark-kind "$kind" --model-id "$model" 2>&1); then
					status=FAILED
					printf '%s\n' "$output" >"$raw_path"
					printf '%s\t%s\t%s\t%s\terror\t%s\t%s\n' "$model" "$loader" "$kind" "$cache_mode" "subprocess_failed" "$status" >>"$SUMMARY"
					continue
				fi
				printf '%s\n' "$output" >"$raw_path"
				python3 - "$raw_path" "$SUMMARY" "$model" "$loader" "$kind" "$cache_mode" "$status" <<'PY'
import json, sys
raw_path, summary, model, loader, kind, cache_mode, status = sys.argv[1:]
with open(raw_path) as f:
    lines = [line.strip() for line in f if line.strip()]
for line in reversed(lines):
    if line.startswith('{'):
        data = json.loads(line)
        break
else:
    with open(summary, 'a') as f:
        f.write(f"{model}\t{loader}\t{kind}\t{cache_mode}\terror\tno_json_payload\tFAILED\n")
    raise SystemExit(0)
with open(summary, 'a') as f:
    for key, value in data.items():
        f.write(f"{model}\t{loader}\t{kind}\t{cache_mode}\t{key}\t{value}\t{status}\n")
PY
			done
		done
	done
done
