"""Unit tests for benchmark vLLM loader dispatch."""

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from benchmarks import vllm_loaders


def test_load_safetensors_dispatch_default(monkeypatch):
    fake = SimpleNamespace(
        load_safetensors=lambda path: {"path": path, "backend": "default"},
        load_safetensors_sync=lambda path: {"path": path, "backend": "sync"},
        load_safetensors_io_uring=lambda path: {"path": path, "backend": "io-uring"},
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_safetensors("/tmp/model", "default")
    assert result["backend"] == "default"


def test_load_safetensors_dispatch_sync(monkeypatch):
    fake = SimpleNamespace(
        load_safetensors=lambda path: {"path": path, "backend": "default"},
        load_safetensors_sync=lambda path: {"path": path, "backend": "sync"},
        load_safetensors_io_uring=lambda path: {"path": path, "backend": "io-uring"},
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_safetensors("/tmp/model", "sync")
    assert result["backend"] == "sync"


def test_load_safetensors_dispatch_io_uring(monkeypatch):
    fake = SimpleNamespace(
        load_safetensors=lambda path: {"path": path, "backend": "default"},
        load_safetensors_sync=lambda path: {"path": path, "backend": "sync"},
        load_safetensors_io_uring=lambda path: {"path": path, "backend": "io-uring"},
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_safetensors("/tmp/model", "io-uring")
    assert result["backend"] == "io-uring"


def test_load_serverlessllm_dispatch_default(monkeypatch):
    fake = SimpleNamespace(
        load_serverlessllm=lambda path: {"path": path, "backend": "default"},
        load_serverlessllm_sync=lambda path: {"path": path, "backend": "sync"},
        load_serverlessllm_io_uring=lambda path: {"path": path, "backend": "io-uring"},
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_serverlessllm("/tmp/model", "default")
    assert result["backend"] == "default"


def test_load_serverlessllm_dispatch_sync(monkeypatch):
    fake = SimpleNamespace(
        load_serverlessllm=lambda path: {"path": path, "backend": "default"},
        load_serverlessllm_sync=lambda path: {"path": path, "backend": "sync"},
        load_serverlessllm_io_uring=lambda path: {"path": path, "backend": "io-uring"},
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_serverlessllm("/tmp/model", "sync")
    assert result["backend"] == "sync"


def test_load_serverlessllm_dispatch_io_uring(monkeypatch):
    fake = SimpleNamespace(
        load_serverlessllm=lambda path: {"path": path, "backend": "default"},
        load_serverlessllm_sync=lambda path: {"path": path, "backend": "sync"},
        load_serverlessllm_io_uring=lambda path: {"path": path, "backend": "io-uring"},
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_serverlessllm("/tmp/model", "io-uring")
    assert result["backend"] == "io-uring"


def test_load_safetensors_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported backend"):
        vllm_loaders._load_safetensors("/tmp/model", "weird")


def test_load_serverlessllm_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported backend"):
        vllm_loaders._load_serverlessllm("/tmp/model", "weird")
