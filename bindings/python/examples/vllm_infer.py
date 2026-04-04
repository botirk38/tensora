#!/usr/bin/env python3
"""Runnable example: run vLLM inference with TensorStore-backed loading.

Usage:
    python examples/vllm_infer.py --model gpt2 --prompt "Hello, world!"

Requirements:
    uv sync --group dev --group vllm
    uv run maturin develop --release
"""

import argparse
import glob
import hashlib
import os
import sys
import warnings


def ensure_serverlessllm(hf_folder: str) -> str:
    from tensor_store_py._tensor_store_rust import convert_safetensors_to_serverlessllm

    safetensors_files = glob.glob(os.path.join(hf_folder, "*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {hf_folder}")

    total_bytes = sum(os.path.getsize(f) for f in safetensors_files)
    parts = max(1, (total_bytes + 512 * 1024 * 1024 - 1) // (512 * 1024 * 1024))

    shard_names = ":".join(sorted(os.path.basename(f) for f in safetensors_files))
    key_input = f"{hf_folder}:main:{parts}:{shard_names}:v3"
    cache_key = hashlib.sha256(key_input.encode()).hexdigest()[:16]

    cache_dir = os.path.join(
        os.path.dirname(hf_folder), ".cache", "serverlessllm", cache_key
    )
    index_path = os.path.join(cache_dir, "tensor_index.json")

    if not os.path.exists(index_path):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Converting to ServerlessLLM format ({parts} partitions)...")
        convert_safetensors_to_serverlessllm(hf_folder, cache_dir, parts)

    return cache_dir


def run_inference(
    model_id: str,
    prompt: str,
    backend: str = "default",
    max_tokens: int = 64,
    temperature: float = 1.0,
) -> str:
    warnings.filterwarnings("ignore")

    from benchmarks.vllm_loaders import register_tensor_store_loader

    register_tensor_store_loader()

    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    print(f"Loading model {model_id} with TensorStore backend={backend}")
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=16384,
        load_format="tensor_store",
        model_loader_extra_config={"format": "serverlessllm", "backend": backend},
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    outputs = llm.generate(prompt, sampling_params=sampling_params)
    return outputs[0].outputs[0].text


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM inference with TensorStore-backed model loading"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--backend",
        default="default",
        choices=["default", "sync", "io-uring"],
        help="TensorStore I/O backend (default: default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )

    args = parser.parse_args()

    try:
        result = run_inference(
            model_id=args.model,
            prompt=args.prompt,
            backend=args.backend,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
