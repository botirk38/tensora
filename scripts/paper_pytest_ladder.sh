#!/usr/bin/env bash
# Reproduce the paper's six public checkpoints (Section "model ladder") via run_benchmarks.sh.
# From the repository root. Requires network for Hugging Face Hub; GPU for vLLM suites.
set -euo pipefail
export TENSORA_BENCH_MODELS="Qwen/Qwen3-0.6B HuggingFaceTB/SmolLM3-3B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/run_benchmarks.sh"
