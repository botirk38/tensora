"""Benchmark-only vLLM tensora loader."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Generator

import torch

from benchmarks.hub_model import recommended_partition_count


def _cache_root() -> str:
    return os.path.join(os.path.dirname(__file__), ".cache", "serverlessllm")


def _load_safetensors(path: str, backend: str) -> dict[str, torch.Tensor]:
    from tensora._tensora_rust import (
        load_safetensors,
        load_safetensors_sync,
    )
    if backend == "io-uring":
        from tensora._tensora_rust import load_safetensors_io_uring

        return load_safetensors_io_uring(path)

    if backend == "default":
        return load_safetensors(path)
    if backend == "sync":
        return load_safetensors_sync(path)
    raise ValueError(f"unsupported backend: {backend}")


def _load_serverlessllm(path: str, backend: str) -> dict[str, torch.Tensor]:
    from tensora._tensora_rust import (
        load_serverlessllm,
        load_serverlessllm_sync,
    )
    if backend == "io-uring":
        from tensora._tensora_rust import load_serverlessllm_io_uring

        return load_serverlessllm_io_uring(path)

    if backend == "default":
        return load_serverlessllm(path)
    if backend == "sync":
        return load_serverlessllm_sync(path)
    raise ValueError(f"unsupported backend: {backend}")


def _ensure_serverlessllm_artifact(
    hf_folder: str,
    safetensors_files: list[str],
    revision: str | None,
) -> str:
    total_bytes = sum(os.path.getsize(path) for path in safetensors_files)
    parts = recommended_partition_count(total_bytes)
    shard_names = ":".join(sorted(os.path.basename(path) for path in safetensors_files))
    key_input = f"{hf_folder}:{revision or 'main'}:{parts}:{shard_names}:v3"
    cache_key = hashlib.sha256(key_input.encode()).hexdigest()[:16]
    out_dir = os.path.join(_cache_root(), cache_key)

    index_path = os.path.join(out_dir, "tensor_index.json")
    if os.path.exists(index_path):
        return out_dir

    os.makedirs(out_dir, exist_ok=True)

    from tensora._tensora_rust import convert_safetensors_to_serverlessllm

    convert_safetensors_to_serverlessllm(hf_folder, out_dir, parts)
    return out_dir


def register_tensora_loader() -> None:
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader("tensora")
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
                raise ValueError(f"unsupported tensora format: {fmt}")
            if backend not in {"default", "sync", "io-uring"}:
                raise ValueError(f"unsupported tensora backend: {backend}")

            original_load_format = self.load_config.load_format
            self.load_config.load_format = "safetensors"
            try:
                hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
                    source.model_or_path,
                    getattr(source, "revision", None),
                    getattr(source, "fall_back_to_pt", False),
                    getattr(source, "allow_patterns_overrides", None),
                )
            finally:
                self.load_config.load_format = original_load_format
            if not use_safetensors:
                raise ValueError(
                    "tensora benchmark loader requires safetensors weights"
                )

            if self.counter_before_loading_weights == 0.0:
                import time

                self.counter_before_loading_weights = time.perf_counter()

            if fmt == "safetensors":
                state_dict = _load_safetensors(hf_folder, backend)
                for name, tensor in state_dict.items():
                    yield source.prefix + name, tensor
                return

            artifact_dir = _ensure_serverlessllm_artifact(
                hf_folder,
                hf_weights_files,
                getattr(source, "revision", None),
            )
            state_dict = _load_serverlessllm(artifact_dir, backend)
            for name, tensor in state_dict.items():
                yield source.prefix + name, tensor
