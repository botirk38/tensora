"""Unit tests for benchmark vLLM loader dispatch."""

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from benchmarks import vllm_loaders


def test_load_safetensors_dispatch_default(monkeypatch):
    fake = SimpleNamespace(
        load_safetensors=lambda path, backend=None: {
            "path": path,
            "backend": backend or "default",
        },
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_safetensors("/tmp/model", "default")
    assert result["backend"] == "default"


def test_load_safetensors_dispatch_sync(monkeypatch):
    fake = SimpleNamespace(
        load_safetensors=lambda path, backend=None: {
            "path": path,
            "backend": backend or "default",
        },
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_safetensors("/tmp/model", "sync")
    assert result["backend"] == "sync"

    def test_load_safetensors_dispatch_io_uring(monkeypatch):
        fake = SimpleNamespace(
            load_safetensors=lambda path, backend=None: {
                "path": path,
                "backend": backend or "default",
            },
        )
        monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
        result = vllm_loaders._load_safetensors("/tmp/model", "io_uring")
        assert result["backend"] == "io_uring"


def test_load_serverlessllm_dispatch_default(monkeypatch):
    fake = SimpleNamespace(
        load_serverlessllm=lambda path, backend=None: {
            "path": path,
            "backend": backend or "default",
        },
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_serverlessllm("/tmp/model", "default")
    assert result["backend"] == "default"


def test_load_serverlessllm_dispatch_sync(monkeypatch):
    fake = SimpleNamespace(
        load_serverlessllm=lambda path, backend=None: {
            "path": path,
            "backend": backend or "default",
        },
    )
    monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
    result = vllm_loaders._load_serverlessllm("/tmp/model", "sync")
    assert result["backend"] == "sync"

    def test_load_serverlessllm_dispatch_io_uring(monkeypatch):
        fake = SimpleNamespace(
            load_serverlessllm=lambda path, backend=None: {
                "path": path,
                "backend": backend or "default",
            },
        )
        monkeypatch.setitem(sys.modules, "tensora._tensora_rust", fake)
        result = vllm_loaders._load_serverlessllm("/tmp/model", "io_uring")
        assert result["backend"] == "io_uring"


def test_load_safetensors_rejects_unknown_backend(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="unsupported backend: weird"):
        vllm_loaders._load_safetensors(str(model_dir), "weird")


def test_load_serverlessllm_rejects_unknown_backend(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="unsupported backend: weird"):
        vllm_loaders._load_serverlessllm(str(model_dir), "weird")
