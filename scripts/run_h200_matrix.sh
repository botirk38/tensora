#!/usr/bin/env bash
set -uo pipefail

source /workspace/env.sh
cd /workspace/tensora || exit 1

MODELS=(
  "Qwen/Qwen3-0.6B"
  "HuggingFaceTB/SmolLM3-3B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-32B"
)
ANCHOR_MODELS=("Qwen/Qwen3-0.6B" "Qwen/Qwen3-8B" "Qwen/Qwen3-32B")
FORMATS=("safetensors" "serverlessllm")
EXPLICIT_BACKENDS=("sync" "mmap" "async" "io-uring")
OUTDIR="/workspace/tensora/results/h200/profile"
PROFILE="./target/release/profile"
EVICT="--evict-page-cache"

mkdir -p "$OUTDIR"

eval "$($PROFILE capabilities --format shell)"

backend_available() {
  case "$1" in
    sync) [[ "${TENSORA_BACKEND_SYNC_AVAILABLE:-false}" == "true" ]] ;;
    mmap) [[ "${TENSORA_BACKEND_MMAP_AVAILABLE:-false}" == "true" ]] ;;
    async) [[ "${TENSORA_BACKEND_ASYNC_AVAILABLE:-false}" == "true" ]] ;;
    io-uring) [[ "${TENSORA_BACKEND_IO_URING_AVAILABLE:-false}" == "true" ]] ;;
    default) return 0 ;;
    *) return 1 ;;
  esac
}

BACKENDS=()
for backend in "${EXPLICIT_BACKENDS[@]}"; do
  if backend_available "$backend"; then
    BACKENDS+=("$backend")
  else
    echo "Skipping unavailable backend: $backend"
  fi
done
BACKENDS+=("default")

record_env() {
  {
    uname -a
    nproc
    free -h
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    awk -F': ' '/model name/ { print $2; exit }' /proc/cpuinfo
    $PROFILE capabilities
  } > "$OUTDIR/env.txt"
}

extract_iteration_field() {
  local output="$1"
  local field="$2"
  local pattern='iteration 1: ([0-9]+) tensors, ([0-9]+) bytes, ([0-9.]+)ms'
  if [[ "$output" =~ $pattern ]]; then
    case "$field" in
      tensors) printf '%s\n' "${BASH_REMATCH[1]}" ;;
      bytes) printf '%s\n' "${BASH_REMATCH[2]}" ;;
      time_ms) printf '%s\n' "${BASH_REMATCH[3]}" ;;
    esac
  fi
}

run_cell() {
  local model="$1"
  local fmt="$2"
  local backend="$3"
  local output

  output=$("$PROFILE" "$fmt" "$backend" --model-id "$model" --iterations 1 $EVICT 2>&1)
  if [[ "$output" != *"iteration 1:"* ]]; then
    printf 'ERROR\t\t\t%s\n' "${output//$'\n'/ }"
    return 0
  fi

  printf '%s\t%s\t%s\t\n' \
    "$(extract_iteration_field "$output" time_ms)" \
    "$(extract_iteration_field "$output" tensors)" \
    "$(extract_iteration_field "$output" bytes)"
}

echo "=== H200 RUST COLD-CACHE MATRIX ==="
date -u
record_env

MATRIX_OUT="$OUTDIR/full_cold_matrix.tsv"
printf 'model\tformat\tbackend\ttime_ms\ttensors\tbytes\terror\n' > "$MATRIX_OUT"

for model in "${MODELS[@]}"; do
  for fmt in "${FORMATS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
      echo "[1x] $model $fmt $backend"
      result=$(run_cell "$model" "$fmt" "$backend")
      printf '%s\t%s\t%s\t%s' "$model" "$fmt" "$backend" "$result" >> "$MATRIX_OUT"
    done
  done
done

ANCHOR_OUT="$OUTDIR/anchor_reps.tsv"
printf 'model\tformat\tbackend\trep\ttime_ms\ttensors\tbytes\terror\n' > "$ANCHOR_OUT"

for model in "${ANCHOR_MODELS[@]}"; do
  for fmt in "${FORMATS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
      for rep in 1 2 3 4 5; do
        echo "[5x rep $rep/5] $model $fmt $backend"
        result=$(run_cell "$model" "$fmt" "$backend")
        printf '%s\t%s\t%s\t%s\t%s' "$model" "$fmt" "$backend" "$rep" "$result" >> "$ANCHOR_OUT"
      done
    done
  done
done

echo "=== MATRIX COMPLETE ==="
date -u
printf 'Results: %s %s\n' "$MATRIX_OUT" "$ANCHOR_OUT"
