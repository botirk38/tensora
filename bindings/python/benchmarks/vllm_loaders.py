"""Benchmark-only vLLM tensor_store loader."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Generator

import torch


def _cache_root() -> str:
    return os.path.join(os.path.dirname(__file__), ".cache", "serverlessllm")


def _partition_count(total_bytes: int) -> int:
    max_parts = min(32, (os.cpu_count() or 4) * 2)
    if total_bytes < 512 * 1024**2:
        parts = 1
    elif total_bytes < 2 * 1024**3:
        parts = 2
    elif total_bytes < 8 * 1024**3:
        parts = 4
    elif total_bytes < 24 * 1024**3:
        parts = 8
    elif total_bytes < 64 * 1024**3:
        parts = 16
    else:
        parts = 32
    return min(parts, max_parts)


def _load_safetensors(path: str, backend: str) -> dict[str, torch.Tensor]:
    from tensor_store_py._tensor_store_rust import (
        load_safetensors,
        load_safetensors_mmap,
        load_safetensors_sync,
    )

    if backend == "default":
        return load_safetensors(path)
    if backend == "sync":
        return load_safetensors_sync(path)
    if backend == "mmap":
        return load_safetensors_mmap(path)
    raise ValueError(f"unsupported backend: {backend}")


def _load_serverlessllm(path: str, backend: str) -> dict[str, torch.Tensor]:
    from tensor_store_py._tensor_store_rust import (
        load_serverlessllm,
        load_serverlessllm_mmap,
        load_serverlessllm_sync,
    )

    if backend == "default":
        return load_serverlessllm(path)
    if backend == "sync":
        return load_serverlessllm_sync(path)
    if backend == "mmap":
        return load_serverlessllm_mmap(path)
    raise ValueError(f"unsupported backend: {backend}")


def _ensure_serverlessllm_artifact(
    hf_folder: str,
    safetensors_files: list[str],
    revision: str | None,
) -> str:
    if len(safetensors_files) != 1:
        raise ValueError(
            "benchmark ServerlessLLM loader currently requires a single safetensors shard"
        )

    total_bytes = sum(os.path.getsize(path) for path in safetensors_files)
    partition_count = _partition_count(total_bytes)
    key_input = f"{hf_folder}:{revision or 'main'}:{partition_count}:v1"
    cache_key = hashlib.sha256(key_input.encode()).hexdigest()[:16]
    out_dir = os.path.join(_cache_root(), cache_key)

    index_path = os.path.join(out_dir, "tensor_index.json")
    if os.path.exists(index_path):
        return out_dir

    os.makedirs(out_dir, exist_ok=True)

    from tensor_store_py._tensor_store_rust import convert_safetensors_to_serverlessllm

    convert_safetensors_to_serverlessllm(safetensors_files[0], out_dir, partition_count)
    return out_dir


def register_tensor_store_loader() -> None:
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader("tensor_store")
    class TensorStoreLoader(DefaultModelLoader):
        def __init__(self, load_config):
            BaseModelLoader.__init__(self, load_config)
            self.local_expert_ids: set[int] | None = None
            self.counter_before_loading_weights = 0.0
            self.counter_after_loading_weights = 0.0

        def _get_weights_iterator(
            self, source: DefaultModelLoader.Source
        ) -> Generator[tuple[str, torch.Tensor], None, None]:
            extra = dict(self.load_config.model_loader_extra_config or {})
            fmt = extra.get("format")
            backend = extra.get("backend")
            if fmt not in {"safetensors", "serverlessllm"}:
                raise ValueError(f"unsupported tensor_store format: {fmt}")
            if backend not in {"default", "sync", "mmap"}:
                raise ValueError(f"unsupported tensor_store backend: {backend}")

            original_load_format = self.load_config.load_format
            self.load_config.load_format = "safetensors"
            try:
                hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
                    source.model_or_path,
                    source.subfolder,
                    source.revision,
                    source.fall_back_to_pt,
                    source.allow_patterns_overrides,
                )
            finally:
                self.load_config.load_format = original_load_format
            if not use_safetensors:
                raise ValueError(
                    "tensor_store benchmark loader requires safetensors weights"
                )

            if self.counter_before_loading_weights == 0.0:
                import time

                self.counter_before_loading_weights = time.perf_counter()

            if fmt == "safetensors":
                for weight_file in sorted(hf_weights_files):
                    state_dict = _load_safetensors(weight_file, backend)
                    for name, tensor in state_dict.items():
                        yield source.prefix + name, tensor
                return

            artifact_dir = _ensure_serverlessllm_artifact(
                hf_folder,
                hf_weights_files,
                source.revision,
            )
            state_dict = _load_serverlessllm(artifact_dir, backend)
            for name, tensor in state_dict.items():
                yield source.prefix + name, tensor
