#!/usr/bin/env bash
# Run paper ladder benchmarks.
# From the repository root. Requires network for Hugging Face Hub.
#
# Usage:
#   TENSORA_BENCH_MODELS="Qwen/Qwen3-0.6B" ./scripts/run_benchmarks.sh    # CPU only
#   TENSORA_BENCH_NO_VLLM=1 TENSORA_BENCH_MODELS="..." ./scripts/run_benchmarks.sh
#
# Defaults:
#   - TENSORA_BENCH_MODELS: required, space-separated HuggingFace model IDs
#   - TENSORA_BENCH_NO_VLLM: if set, skip vLLM benchmarks (CPU-only)
#   - TENSORA_BENCH_JSON: output directory for JSON files (default: <repo>/results/benchmarks/)

set -euo pipefail
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
py_dir="${repo_root}/bindings/python"

if [[ -z "${TENSORA_BENCH_MODELS:-}" ]]; then
  models=("Qwen/Qwen3-0.6B" "HuggingFaceTB/SmolLM3-3B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B")
else
  models=(${TENSORA_BENCH_MODELS})
fi

if [[ "${TENSORA_BENCH_NO_VLLM:-}" == "1" ]]; then
  echo "==> uv sync (dev + torch; skipping vLLM)"
  uv --directory "${py_dir}" sync --group dev --group torch
  tests=(benchmarks/bench_safetensors.py benchmarks/bench_serverlessllm.py)
else
  echo "==> uv sync (dev + vllm)"
  uv --directory "${py_dir}" sync --group dev --group vllm
  tests=(benchmarks/bench_safetensors.py benchmarks/bench_serverlessllm.py benchmarks/bench_vllm.py)
fi

echo "==> maturin develop --release"
uv --directory "${py_dir}" run maturin develop --release

if [[ -n "${TENSORA_BENCH_JSON:-}" ]]; then
  out_dir="${TENSORA_BENCH_JSON}"
else
  out_dir="${repo_root}/results/benchmarks"
fi
mkdir -p "${out_dir}"

for model_id in "${models[@]}"; do
  slug="${model_id//\//-}"
  slug="${slug,,}"
  bench_json="${out_dir}/pytest_benchmark_${slug}.json"
  cache_dir="${out_dir}/.cache/${slug}"
  mkdir -p "${cache_dir}"

  echo ""
  echo "==> Running safetensors benchmarks for ${model_id}"

  if ! uv --directory "${py_dir}" run pytest \
    benchmarks/bench_safetensors.py \
    -v \
    --model-id "${model_id}" \
    --cache-dir "${cache_dir}" \
    --benchmark-json="${bench_json}.safetensors" \
    --benchmark-disable-gc \
    --benchmark-min-rounds=1; then
    echo "WARNING: safetensors tests failed for ${model_id}; continuing"
  fi

  echo ""
  echo "==> Running serverlessllm benchmarks for ${model_id}"

  if ! uv --directory "${py_dir}" run pytest \
    benchmarks/bench_serverlessllm.py \
    -v \
    --model-id "${model_id}" \
    --cache-dir "${cache_dir}" \
    --benchmark-json="${bench_json}.serverlessllm" \
    --benchmark-disable-gc \
    --benchmark-min-rounds=1; then
    echo "WARNING: serverlessllm tests failed for ${model_id}; continuing"
  fi

  if [[ "${TENSORA_BENCH_NO_VLLM:-}" != "1" ]]; then
    echo ""
    echo "==> Running vLLM benchmarks for ${model_id}"

    vllm_pytest_args=()
    if [[ -n "${TENSORA_BENCH_SKIP_LOADERS:-}" ]]; then
      vllm_pytest_args+=("-k" "not (${TENSORA_BENCH_SKIP_LOADERS})")
    fi

    if ! uv --directory "${py_dir}" run pytest \
      benchmarks/bench_vllm.py \
      -v \
      "${vllm_pytest_args[@]}" \
      --model-id "${model_id}" \
      --cache-dir "${cache_dir}" \
      --benchmark-json="${bench_json}.vllm" \
      --benchmark-min-rounds=1; then
      echo "WARNING: vLLM tests failed for ${model_id}; continuing"
    fi
  fi

  echo "==> Completed ${model_id}"
done

echo ""
echo "Done. Results in ${out_dir}/"
ls -la "${out_dir}/pytest_benchmark_"*.json* 2>/dev/null || true