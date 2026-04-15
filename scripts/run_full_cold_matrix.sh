#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RESULTS="results/h100/profile/full_cold_matrix.tsv"
mkdir -p "$(dirname "$RESULTS")"
echo -e "format\tmodel\tbackend\titer1_ms\tmean_ms\tmin_ms\tmax_ms\ttensors\tbytes\tstatus" >"$RESULTS"

FORMATS=(safetensors serverlessllm)
FIXTURES=(qwen-qwen3-0.6b huggingfacetb-smollm3-3b qwen-qwen3-4b qwen-qwen3-8b qwen-qwen3-14b qwen-qwen3-32b)
BACKENDS=(sync async default io-uring)

for format in "${FORMATS[@]}"; do
  for fixture in "${FIXTURES[@]}"; do
    for backend in "${BACKENDS[@]}"; do
      echo "=== COLD $format $fixture $backend ==="
      sync
      echo 3 >/proc/sys/vm/drop_caches
      output=$(./target/release/profile "$format" "$backend" --fixture "$fixture" --iterations 1 2>&1) || true

      summary=$(printf '%s\n' "$output" | python3 -c 'import re, sys; text=sys.stdin.read(); m=re.search(r"summary: mean ([0-9.]+)ms \| min ([0-9.]+)ms \| max ([0-9.]+)ms", text); print("\t".join(m.groups()) if m else "")')
      iter_line=$(printf '%s\n' "$output" | grep 'iteration 1:' | head -1 || true)
      iter_ms=$(printf '%s\n' "$iter_line" | python3 -c 'import re, sys; text=sys.stdin.read(); m=re.search(r"([0-9.]+)ms", text); print(m.group(1) if m else "")')
      tensors=$(printf '%s\n' "$iter_line" | python3 -c 'import re, sys; text=sys.stdin.read(); m=re.search(r"(\d+) tensors", text); print(m.group(1) if m else "")')
      bytes=$(printf '%s\n' "$iter_line" | python3 -c 'import re, sys; text=sys.stdin.read(); m=re.search(r"(\d+) bytes", text); print(m.group(1) if m else "")')

      if [ -z "$summary" ] || [ -z "$iter_ms" ]; then
        echo -e "$format\t$fixture\t$backend\t\t\t\t\t\t\tFAILED" >>"$RESULTS"
        echo "  FAILED"
        continue
      fi

      echo -e "$format\t$fixture\t$backend\t$iter_ms\t$summary\t$tensors\t$bytes\tOK" >>"$RESULTS"
      echo "  iter=${iter_ms}ms summary=${summary} tensors=$tensors bytes=$bytes"
    done
  done
done

echo "Results saved to $RESULTS"
