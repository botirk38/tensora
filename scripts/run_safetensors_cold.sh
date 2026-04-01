#!/bin/bash
set -euo pipefail

cd ~/PROJECT

RESULTS="results/h100/profile/safetensors_cold.tsv"
echo "format	model	backend	iter1_ms	mean_ms	min_ms	max_ms	tensors	bytes" >"$RESULTS"

FIXTURES=(qwen-qwen3-0.6b huggingfacetb-smollm3-3b qwen-qwen3-4b qwen-qwen3-8b qwen-qwen3-14b qwen-qwen3-32b)
BACKENDS=(sync async default io-uring)

for fixture in "${FIXTURES[@]}"; do
	for backend in "${BACKENDS[@]}"; do
		echo "=== COLD $fixture $backend ==="
		sync
		echo 3 >/proc/sys/vm/drop_caches
		output=$(./target/release/profile safetensors "$backend" --fixture "$fixture" --iterations 1 2>&1) || true

		summary=$(printf '%s\n' "$output" | python3 -c 'import re, sys; text=sys.stdin.read(); match=re.search(r"summary: mean ([0-9.]+)ms \| min ([0-9.]+)ms \| max ([0-9.]+)ms", text); print("\t".join(match.groups()) if match else "")')
		if [ -z "$summary" ]; then
			echo "FAILED: safetensors $fixture $backend" >>"$RESULTS"
			echo "  FAILED"
			continue
		fi

		iter_line=$(printf '%s\n' "$output" | grep 'iteration 1:' | head -1)
		iter_ms=$(printf '%s\n' "$iter_line" | python3 -c 'import re, sys; text=sys.stdin.read(); match=re.search(r"([0-9.]+)ms", text); print(match.group(1) if match else "")')
		tensors=$(printf '%s\n' "$iter_line" | python3 -c 'import re, sys; text=sys.stdin.read(); match=re.search(r"(\d+) tensors", text); print(match.group(1) if match else "")')
		bytes=$(printf '%s\n' "$iter_line" | python3 -c 'import re, sys; text=sys.stdin.read(); match=re.search(r"(\d+) bytes", text); print(match.group(1) if match else "")')

		echo -e "safetensors\t$fixture\t$backend\t$iter_ms\t$summary\t$tensors\t$bytes" >>"$RESULTS"
		echo "  iter=${iter_ms}ms summary=${summary} tensors=$tensors bytes=$bytes"
	done
done

echo "Results saved to $RESULTS"
