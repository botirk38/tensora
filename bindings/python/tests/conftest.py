"""Shared pytest fixtures for tensora tests."""

import pytest

from tensora._tensora_rust import (
    load_safetensors,
    load_safetensors_async,
    load_safetensors_sync,
    load_serverlessllm,
    load_serverlessllm_async,
    load_serverlessllm_sync,
    open_safetensors,
    open_serverlessllm,
)
from tests.fixtures import (
    create_gpt2,
    write_safetensors,
    write_safetensors_dir,
    write_serverlessllm_dir,
)


def get_keys_sync(path, fmt):
    if fmt == "safetensors":
        d = load_safetensors_sync(path)
    else:
        d = load_serverlessllm_sync(path)
    return set(d.keys())


def get_keys_open(path, fmt):
    if fmt == "safetensors":
        d = open_safetensors(path)
    else:
        d = open_serverlessllm(path)
    return set(d.keys())


def get_keys_default(path, fmt):
    if fmt == "safetensors":
        d = load_safetensors(path)
    else:
        d = load_serverlessllm(path)
    return set(d.keys())


def get_shapes_sync(path, fmt):
    if fmt == "safetensors":
        d = load_safetensors_sync(path)
    else:
        d = load_serverlessllm_sync(path)
    return {k: tuple(d[k].shape) for k in sorted(d.keys())}


def get_shapes_open(path, fmt):
    if fmt == "safetensors":
        d = open_safetensors(path)
    else:
        d = open_serverlessllm(path)
    return {k: tuple(d[k].shape) for k in sorted(d.keys())}


def get_shapes_default(path, fmt):
    if fmt == "safetensors":
        d = load_safetensors(path)
    else:
        d = load_serverlessllm(path)
    return {k: tuple(d[k].shape) for k in sorted(d.keys())}


@pytest.fixture
def safetensors_path(tmp_path):
    """GPT-2-like SafeTensors model directory (2 layers)."""
    tensors = create_gpt2(n_layers=2)
    return str(write_safetensors_dir(tensors, tmp_path / "model_dir"))


@pytest.fixture
def hidden_dim():
    """Hidden dimension used by create_gpt2 fixture."""
    return 126


@pytest.fixture
def seq_len():
    """Sequence length used by create_gpt2 fixture."""
    return 128


@pytest.fixture
def serverlessllm_dir(tmp_path):
    """GPT-2-like ServerlessLLM directory (2 layers)."""
    tensors = create_gpt2(n_layers=2)
    out_dir = write_serverlessllm_dir(tensors, tmp_path / "model_sllm")
    return str(out_dir)


@pytest.fixture
def serverlessllm_dir_4partitions(tmp_path):
    """GPT-2-like ServerlessLLM directory with 4 partitions."""
    tensors = create_gpt2(n_layers=2)
    out_dir = write_serverlessllm_dir(
        tensors, tmp_path / "model_sllm_4p", num_partitions=4
    )
    return str(out_dir)


@pytest.fixture
def serverlessllm_dir_8partitions(tmp_path):
    """GPT-2-like ServerlessLLM directory with 8 partitions."""
    tensors = create_gpt2(n_layers=2)
    out_dir = write_serverlessllm_dir(
        tensors, tmp_path / "model_sllm_8p", num_partitions=8
    )
    return str(out_dir)


@pytest.fixture
def safetensors_path_small(tmp_path):
    """Small SafeTensors model directory (single tensor)."""
    import torch

    tensors = {"x": torch.randn(2, 3)}
    return str(write_safetensors_dir(tensors, tmp_path / "model_dir"))


@pytest.fixture
def serverlessllm_dir_small(tmp_path):
    """Small ServerlessLLM directory (single tensor)."""
    import torch

    tensors = {"x": torch.randn(2, 3)}
    out_dir = write_serverlessllm_dir(tensors, tmp_path / "model_sllm")
    return str(out_dir)


@pytest.fixture
def safetensors_path_dtypes(tmp_path):
    """SafeTensors model directory with multiple dtypes for roundtrip checks."""
    import torch

    tensors = {
        "f32": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        "i64": torch.tensor([1, 2, 3], dtype=torch.int64),
        "bool": torch.tensor([True, False, True], dtype=torch.bool),
        "scalar": torch.tensor(42.0, dtype=torch.float32),
        "empty": torch.zeros(0, 2),
    }
    return str(write_safetensors_dir(tensors, tmp_path / "dtypes_dir"))


@pytest.fixture
def safetensors_path_large(tmp_path):
    """GPT-2-like SafeTensors model directory (12 layers) for heavier integration tests."""
    tensors = create_gpt2(n_layers=12)
    return str(write_safetensors_dir(tensors, tmp_path / "model_large_dir"))


@pytest.fixture
def serverlessllm_dir_large(tmp_path):
    """GPT-2-like ServerlessLLM directory (12 layers) for heavier integration tests."""
    tensors = create_gpt2(n_layers=12)
    out_dir = write_serverlessllm_dir(tensors, tmp_path / "model_sllm_large")
    return str(out_dir)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration (run with tests/integration/)",
    )
