#!/bin/bash
set -euo pipefail

cd ~/PROJECT

RESULTS="results/h100/profile/anchor_reps.tsv"
mkdir -p "$(dirname "$RESULTS")"
echo "format	model	backend	rep	ms	tensors	bytes" >"$RESULTS"

FIXTURES=(qwen-qwen3-0.6b qwen-qwen3-8b qwen-qwen3-32b)
FORMATS=(safetensors serverlessllm)
BACKENDS=(sync async io-uring default)
REPS=5

for format in "${FORMATS[@]}"; do
	for fixture in "${FIXTURES[@]}"; do
		for backend in "${BACKENDS[@]}"; do
			echo "=== COLD $format $fixture $backend (${REPS} reps) ==="
			for rep in $(seq 1 $REPS); do
				sync
				echo 3 >/proc/sys/vm/drop_caches
				output=$(./target/release/profile "$format" "$backend" --fixture "$fixture" --iterations 1 2>&1) || true

				iter_line=$(printf '%s\n' "$output" | grep 'iteration 1:' | head -1)
				ms=$(printf '%s\n' "$iter_line" | python3 -c 'import re, sys; text=sys.stdin.read(); match=re.search(r"([0-9.]+)ms", text); print(match.group(1) if match else "")')
				tensors=$(printf '%s\n' "$iter_line" | python3 -c 'import re, sys; text=sys.stdin.read(); match=re.search(r"(\d+) tensors", text); print(match.group(1) if match else "")')
				bytes=$(printf '%s\n' "$iter_line" | python3 -c 'import re, sys; text=sys.stdin.read(); match=re.search(r"(\d+) bytes", text); print(match.group(1) if match else "")')

				if [ -z "$ms" ]; then
					echo "FAILED: $format $fixture $backend rep $rep" >>"$RESULTS"
					echo "  rep $rep: FAILED"
				else
					echo -e "$format\t$fixture\t$backend\t$rep\t$ms\t$tensors\t$bytes" >>"$RESULTS"
					echo "  rep $rep: ${ms}ms"
				fi
			done
		done
	done
done

echo ""
echo "Results saved to $RESULTS"
