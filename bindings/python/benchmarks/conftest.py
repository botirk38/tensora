"""Pytest fixtures for benchmarks: synthetic SafeTensors and ServerlessLLM paths."""

import pytest

from benchmarks.fixtures import create_gpt2, write_safetensors, write_serverlessllm_dir


@pytest.fixture(scope="session")
def safetensors_path(tmp_path_factory):
    """Session-scoped path to a synthetic SafeTensors file (GPT-2-like)."""
    tmp = tmp_path_factory.mktemp("bench_fixture")
    tensors = create_gpt2(n_layers=2)
    return str(write_safetensors(tensors, tmp / "model.safetensors"))


@pytest.fixture(scope="session")
def serverlessllm_dir(tmp_path_factory):
    """Session-scoped path to a synthetic ServerlessLLM directory (GPT-2-like)."""
    tmp = tmp_path_factory.mktemp("bench_fixture")
    tensors = create_gpt2(n_layers=2)
    return str(write_serverlessllm_dir(tensors, tmp / "model_sllm"))
