"""vLLM benchmark runner: subprocess entrypoint for vLLM benchmarks.

Usage:
    python -m benchmarks.vllm_runner --loader native --benchmark-kind ttft --model-id gpt2

Output is machine-readable JSON for parsing by benchmark harness.
"""

import argparse
import json
import logging
import os
import sys
import time


BENCHMARK_KINDS = ["load_only", "ttft", "steady_state_decode"]
LOADERS = [
    "native",
    "ts_safetensors_default",
    "ts_safetensors_sync",
    "ts_safetensors_io_uring",
    "ts_serverlessllm_default",
    "ts_serverlessllm_sync",
    "ts_serverlessllm_io_uring",
]

LOADER_CONFIG = {
    "native": None,
    "ts_safetensors_default": {"format": "safetensors", "backend": "default"},
    "ts_safetensors_sync": {"format": "safetensors", "backend": "sync"},
    "ts_safetensors_io_uring": {"format": "safetensors", "backend": "io_uring"},
    "ts_serverlessllm_default": {"format": "serverlessllm", "backend": "default"},
    "ts_serverlessllm_sync": {"format": "serverlessllm", "backend": "sync"},
    "ts_serverlessllm_io_uring": {"format": "serverlessllm", "backend": "io_uring"},
}


def get_llm_kwargs(model_id: str, loader: str = None) -> dict:
    """Get vLLM engine kwargs, with model-specific memory settings."""
    base_kwargs = {
        "model": model_id,
        "tensor_parallel_size": 1,
        "max_model_len": 16384,
        "enforce_eager": True,
    }

    is_custom = loader and loader.startswith("ts_")

    if "8B" in model_id or "8b" in model_id:
        base_kwargs["gpu_memory_utilization"] = 0.70
    elif "14B" in model_id or "14b" in model_id:
        base_kwargs["gpu_memory_utilization"] = 0.50
    elif "32B" in model_id or "32b" in model_id:
        base_kwargs["gpu_memory_utilization"] = 0.95
        base_kwargs["max_model_len"] = 16384
    elif (
        "72B" in model_id or "72b" in model_id or "70B" in model_id or "70b" in model_id
    ):
        base_kwargs["gpu_memory_utilization"] = 0.15
    else:
        base_kwargs["gpu_memory_utilization"] = 0.95

    return base_kwargs


def _cleanup_engine(llm: object) -> None:
    """Shut down a vLLM LLM instance and release GPU memory."""
    import gc

    import torch

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_benchmark(
    model_id: str,
    loader: str,
    benchmark_kind: str,
    num_tokens: int = 1,
) -> dict:
    """Run vLLM benchmark and return timing results."""
    import warnings

    warnings.filterwarnings("ignore")

    logging.getLogger("vllm").setLevel(logging.CRITICAL)
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    logging.getLogger("accelerate").setLevel(logging.CRITICAL)

    os.environ["VLLM_LOGGING_LEVEL"] = "CRITICAL"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = LOADER_CONFIG.get(loader)

    if config is not None:
        from benchmarks.vllm_loaders import register_tensora_loader

        register_tensora_loader()

    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    llm_kwargs = get_llm_kwargs(model_id, loader)

    if config is not None:
        llm_kwargs["load_format"] = "tensora"
        llm_kwargs["model_loader_extra_config"] = config

    init_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    init_end = time.perf_counter()
    init_ms = (init_end - init_start) * 1000
    results = {"init_ms": init_ms}

    try:
        if benchmark_kind == "load_only":
            return results

        if benchmark_kind == "ttft":
            generation_start = time.perf_counter()
            sampling_params = SamplingParams(max_tokens=num_tokens)
            _ = llm.generate("Hello, my name is", sampling_params=sampling_params)
            generation_end = time.perf_counter()
            first_token_ms = (generation_end - generation_start) * 1000
            results["first_token_ms"] = first_token_ms
            results["ttft_ms"] = init_ms + first_token_ms

        if benchmark_kind == "steady_state_decode":
            warmup_params = SamplingParams(max_tokens=8)
            _ = llm.generate(
                "The quick brown fox jumps over the lazy dog.", warmup_params
            )

            decode_times = []
            for _ in range(5):
                decode_start = time.perf_counter()
                sampling_params = SamplingParams(max_tokens=8)
                _ = llm.generate("Hello", sampling_params=sampling_params)
                decode_end = time.perf_counter()
                decode_times.append((decode_end - decode_start) * 1000)

            results["decode_avg_ms"] = sum(decode_times) / len(decode_times)
            results["decode_min_ms"] = min(decode_times)
            results["decode_max_ms"] = max(decode_times)

        return results
    finally:
        _cleanup_engine(llm)


def main():
    parser = argparse.ArgumentParser(description="vLLM benchmark runner")
    parser.add_argument(
        "--loader",
        required=True,
        choices=LOADERS,
        help="Which loader to use",
    )
    parser.add_argument(
        "--benchmark-kind",
        type=str,
        default="ttft",
        choices=BENCHMARK_KINDS,
        help="Type of benchmark to run",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID (e.g., gpt2)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1,
        help="Number of tokens to generate",
    )
    args = parser.parse_args()

    try:
        results = run_benchmark(
            model_id=args.model_id,
            loader=args.loader,
            benchmark_kind=args.benchmark_kind,
            num_tokens=args.num_tokens,
        )

        print(json.dumps(results))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
