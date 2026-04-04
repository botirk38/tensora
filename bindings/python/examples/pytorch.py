#!/usr/bin/env python3
"""Load safetensors with PyTorch using TensorStore and run inference.

Usage:
    python examples/pytorch.py gpt2 --prompt "Hello, world!"

Requirements:
    uv sync --group dev --group torch
    uv run maturin develop --release
"""

import argparse
import glob
import hashlib
import os
import sys
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer


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
    fmt: str = "safetensors",
    backend: str = "default",
    device: str = "cuda",
    max_tokens: int = 64,
    temperature: float = 1.0,
) -> str:
    warnings.filterwarnings("ignore")

    print(f"Downloading model {model_id}...")
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(model_id, local_files_only=False)

    if fmt == "serverlessllm":
        local_path = ensure_serverlessllm(local_path)
        print(f"Using serverlessllm format")
    else:
        print(f"Using safetensors format")

    print(f"Loading weights with TensorStore backend={backend}")
    if fmt == "serverlessllm":
        if backend == "sync":
            from tensor_store_py._tensor_store_rust import load_serverlessllm_sync

            state_dict = load_serverlessllm_sync(local_path)
        elif backend == "io-uring":
            from tensor_store_py._tensor_store_rust import load_serverlessllm_io_uring

            state_dict = load_serverlessllm_io_uring(local_path)
        else:
            from tensor_store_py._tensor_store_rust import load_serverlessllm

            state_dict = load_serverlessllm(local_path)
    else:
        if backend == "sync":
            from tensor_store_py.torch import load_safetensors_sync

            state_dict = load_safetensors_sync(local_path, device=device)
        elif backend == "io-uring":
            from tensor_store_py._tensor_store_rust import load_safetensors_io_uring

            state_dict = load_safetensors_io_uring(
                local_path, framework="torch", device=device
            )
        else:
            from tensor_store_py.torch import load_safetensors

            state_dict = load_safetensors(local_path, device=device)

    print(f"Initializing model with {len(state_dict)} tensors...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    model.load_state_dict(state_dict, strict=False)

    print(f"Running inference...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Load safetensors with PyTorch using TensorStore and run inference"
    )
    parser.add_argument(
        "model",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--format",
        default="safetensors",
        choices=["safetensors", "serverlessllm"],
        help="TensorStore checkpoint format (default: safetensors)",
    )
    parser.add_argument(
        "--backend",
        default="default",
        choices=["default", "sync", "io-uring"],
        help="TensorStore I/O backend (default: default)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to load tensors to (default: cuda)",
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
            fmt=args.format,
            backend=args.backend,
            device=args.device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
