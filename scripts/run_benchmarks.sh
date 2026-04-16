#!/usr/bin/env bash
# Run pytest benchmarks under bindings/python/benchmarks via uv.
# Writes pytest-benchmark JSON (--benchmark-json), one file per model.
#
# Required:
#   TENSORA_BENCH_MODELS   One or more Hugging Face model ids (space- and/or comma-separated)
#
# Optional:
#   TENSORA_BENCH_JSON          Output directory for JSON files (default: <repo>/results/benchmarks/)
#   TENSORA_BENCH_JOBS          Max concurrent pytest processes (default: min(4, number of models); ignored when vLLM runs)
#   TENSORA_SKIP_MAURIN=1       Skip maturin develop --release
#   TENSORA_BENCH_NO_VLLM=1     Omit bench_vllm.py; uv sync uses --group dev --group torch (no vLLM stack)

set -euo pipefail

# Non-login shells and minimal images often omit these; maturin needs rustc, uv may live in ~/.local/bin.
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
if git -C "${repo_root}" rev-parse --show-toplevel &>/dev/null; then
  repo_root="$(git -C "${repo_root}" rev-parse --show-toplevel)"
fi

py_dir="${repo_root}/bindings/python"

if [[ -z "${TENSORA_BENCH_MODELS:-}" ]]; then
  echo "error: set TENSORA_BENCH_MODELS to one or more Hugging Face model ids (space/comma-separated; example: gpt2 or Qwen/Qwen3-8B)" >&2
  exit 1
fi

_normalized="${TENSORA_BENCH_MODELS//,/ }"
read -r -a models <<< "${_normalized}"
if [[ ${#models[@]} -eq 0 ]]; then
  echo "error: TENSORA_BENCH_MODELS produced no model ids after parsing" >&2
  exit 1
fi

if [[ -n "${TENSORA_BENCH_JSON:-}" ]]; then
  if [[ -f "${TENSORA_BENCH_JSON}" ]]; then
    echo "error: TENSORA_BENCH_JSON must be a directory path, not an existing file (${TENSORA_BENCH_JSON})" >&2
    exit 1
  fi
  if [[ "${TENSORA_BENCH_JSON}" == *.json ]] && [[ ! -d "${TENSORA_BENCH_JSON}" ]]; then
    echo "error: TENSORA_BENCH_JSON must be a directory; remove the .json suffix or create the directory" >&2
    exit 1
  fi
  out_dir="${TENSORA_BENCH_JSON}"
else
  out_dir="${repo_root}/results/benchmarks"
fi
mkdir -p "${out_dir}"

slugify() {
  local s="${1//\//-}"
  echo "${s,,}"
}

n_models=${#models[@]}

if [[ "${TENSORA_BENCH_NO_VLLM:-}" == "1" ]]; then
  echo "==> uv sync (dev + torch; skipping vLLM dependency group)"
  uv --directory "${py_dir}" sync --group dev --group torch
else
  echo "==> uv sync (dev + vllm)"
  uv --directory "${py_dir}" sync --group dev --group vllm
fi

if [[ "${TENSORA_SKIP_MAURIN:-}" != "1" ]]; then
  echo "==> maturin develop --release"
  uv --directory "${py_dir}" run maturin develop --release
else
  echo "==> skipping maturin (TENSORA_SKIP_MAURIN=1)"
fi

tests=(benchmarks/bench_safetensors.py benchmarks/bench_serverlessllm.py)
if [[ "${TENSORA_BENCH_NO_VLLM:-}" != "1" ]]; then
  tests+=(benchmarks/bench_vllm.py)
else
  echo "==> skipping bench_vllm.py (TENSORA_BENCH_NO_VLLM=1)"
fi

if [[ "${TENSORA_BENCH_NO_VLLM:-}" != "1" ]]; then
  effective_jobs=1
  if [[ -n "${TENSORA_BENCH_JOBS:-}" ]] && [[ "${TENSORA_BENCH_JOBS}" != "1" ]]; then
    echo "==> vLLM benchmarks: forcing parallel jobs to 1 (TENSORA_BENCH_JOBS=${TENSORA_BENCH_JOBS} ignored; single GPU / single vLLM process)" >&2
  fi
else
  if [[ -n "${TENSORA_BENCH_JOBS:-}" ]]; then
    effective_jobs="${TENSORA_BENCH_JOBS}"
  else
    effective_jobs=$n_models
    (( effective_jobs > 4 )) && effective_jobs=4
  fi
  (( effective_jobs < 1 )) && effective_jobs=1
  (( effective_jobs > n_models )) && effective_jobs=$n_models
fi

bench_any_fail=0

run_pytest_model() {
  local model_id="$1"
  local slug
  slug="$(slugify "${model_id}")"
  local bench_json_path="${out_dir}/pytest_benchmark_${slug}.json"
  local cache_dir="${out_dir}/.cache/${slug}"
  mkdir -p "${cache_dir}"
  echo "==> pytest model=${model_id} -> ${bench_json_path} (cache: ${cache_dir})"
  if ! uv --directory "${py_dir}" run pytest "${tests[@]}" -v \
    --model-id "${model_id}" \
    --cache-dir "${cache_dir}" \
    --benchmark-json="${bench_json_path}"; then
    echo "warning: pytest exited non-zero for model ${model_id}; continuing with remaining models" >&2
    return 1
  fi
  return 0
}

if (( effective_jobs == 1 )); then
  for model_id in "${models[@]}"; do
    run_pytest_model "${model_id}" || bench_any_fail=1
  done
else
  echo "==> running up to ${effective_jobs} pytest job(s) in parallel (CPU-only benchmarks)"
  idx=0
  while (( idx < n_models )); do
    batch_pids=()
    for (( j = 0; j < effective_jobs && idx < n_models; j++, idx++ )); do
      model_id="${models[idx]}"
      (
        run_pytest_model "${model_id}"
      ) &
      batch_pids+=($!)
    done
    for pid in "${batch_pids[@]}"; do
      if ! wait "${pid}"; then
        bench_any_fail=1
      fi
    done
  done
fi

echo "Done. JSON under ${out_dir}/pytest_benchmark_<slug>.json"
exit "${bench_any_fail}"
