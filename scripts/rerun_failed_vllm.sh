#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/bindings/python"
RESULTS_DIR="$REPO_ROOT/results/h100/vllm"
SUMMARY="$RESULTS_DIR/full_matrix.tsv"
mkdir -p "$RESULTS_DIR/raw"

python3 - <<'PY' "$SUMMARY" > /tmp/vllm_failed_cases.tsv
import csv, sys
summary = sys.argv[1]
seen = set()
with open(summary) as f:
    rows = csv.DictReader(f, delimiter='\t')
    for row in rows:
        if row['status'] == 'FAILED':
            key = (row['model'], row['loader'], row['kind'], row['cache_mode'])
            if key not in seen:
                seen.add(key)
                print('\t'.join(key))
PY

while IFS=$'\t' read -r model loader kind cache_mode; do
  [ -z "$model" ] && continue
  echo "=== RETRY $model $cache_mode $loader $kind ==="
  if [ "$cache_mode" = "cold" ]; then
    sync
    echo 3 >/proc/sys/vm/drop_caches
  fi
  model_slug=$(echo "$model" | tr '[:upper:]' '[:lower:]' | sed 's|/|-|g; s|[^a-z0-9._-]|-|g')
  raw_path="$RESULTS_DIR/raw/${model_slug}_${loader}_${kind}_${cache_mode}.log"
  if output=$(.venv/bin/python -m benchmarks.vllm_runner --loader "$loader" --benchmark-kind "$kind" --model-id "$model" 2>&1); then
    printf '%s\n' "$output" > "$raw_path"
    python3 - <<'PY' "$SUMMARY" "$raw_path" "$model" "$loader" "$kind" "$cache_mode"
import csv, json, sys
summary, raw_path, model, loader, kind, cache_mode = sys.argv[1:]
with open(raw_path) as f:
    lines=[line.strip() for line in f if line.strip()]
for line in reversed(lines):
    if line.startswith('{'):
        data=json.loads(line)
        break
else:
    raise SystemExit('no json payload')
with open(summary) as f:
    rows=list(csv.reader(f, delimiter='\t'))
header, body = rows[0], rows[1:]
body=[r for r in body if not (len(r)>=7 and r[0]==model and r[1]==loader and r[2]==kind and r[3]==cache_mode)]
for key, value in data.items():
    body.append([model, loader, kind, cache_mode, key, str(value), 'OK'])
with open(summary, 'w', newline='') as f:
    writer=csv.writer(f, delimiter='\t')
    writer.writerow(header)
    writer.writerows(body)
PY
  else
    printf '%s\n' "$output" > "$raw_path"
  fi
done < /tmp/vllm_failed_cases.tsv
