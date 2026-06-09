"""Type-safe enumerations for the experiment domain.

Every dimension of the experiment matrix (format, backend, loader,
benchmark kind) is a StrEnum so it serialises cleanly to CLI args
and TSV columns while making typos a static-analysis error.
"""

from __future__ import annotations

from enum import StrEnum


class Format(StrEnum):
    """Checkpoint layout used during profiling."""

    SAFETENSORS = "safetensors"
    SERVERLESSLLM = "serverlessllm"


class Backend(StrEnum):
    """I/O submission backend for the Rust profiler."""

    SYNC = "sync"
    ASYNC = "async"
    IO_URING = "io-uring"
    DEFAULT = "default"
    HF_NATIVE = "hf-native"


class BenchmarkKind(StrEnum):
    """What the vLLM harness measures for a single run."""

    LOAD_ONLY = "load_only"
    TTFT = "ttft"
    STEADY_STATE_DECODE = "steady_state_decode"


class VllmLoader(StrEnum):
    """Weight-loader variant passed to the vLLM runner script."""

    NATIVE = "native"
    TS_SAFETENSORS_SYNC = "ts_safetensors_sync"
    TS_SAFETENSORS_ASYNC = "ts_safetensors_async"
    TS_SERVERLESSLLM_SYNC = "ts_serverlessllm_sync"
    TS_SERVERLESSLLM_ASYNC = "ts_serverlessllm_async"

    @property
    def tensora_config(self) -> dict[str, str] | None:
        """Return the (format, backend) pair for Tensora loaders, None for native."""
        mapping: dict[VllmLoader, dict[str, str] | None] = {
            VllmLoader.NATIVE: None,
            VllmLoader.TS_SAFETENSORS_SYNC: {
                "format": Format.SAFETENSORS,
                "backend": Backend.SYNC,
            },
            VllmLoader.TS_SAFETENSORS_ASYNC: {
                "format": Format.SAFETENSORS,
                "backend": Backend.ASYNC,
            },
            VllmLoader.TS_SERVERLESSLLM_SYNC: {
                "format": Format.SERVERLESSLLM,
                "backend": Backend.SYNC,
            },
            VllmLoader.TS_SERVERLESSLLM_ASYNC: {
                "format": Format.SERVERLESSLLM,
                "backend": Backend.ASYNC,
            },
        }
        return mapping[self]
