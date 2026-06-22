"""Pytest configuration for benchmarks: CLI options and shared setup."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--model-id",
        action="store",
        required=True,
        help="HuggingFace model ID (e.g., gpt2, Qwen/Qwen2-0.5B)",
    )
    parser.addoption(
        "--cache-dir",
        action="store",
        default=None,
        help="Directory to store downloaded models and conversion artifacts",
    )


@pytest.fixture(scope="session")
def model_id(request):
    """Model ID from command line."""
    return request.config.getoption("--model-id")


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory, request):
    """Base directory for model downloads and cached artifacts."""
    user_dir = request.config.getoption("--cache-dir")
    if user_dir:
        return Path(user_dir)
    return tmp_path_factory.mktemp("bench_cache")


@pytest.fixture(scope="session")
def model_descriptor(model_id, cache_dir):
    """Full metadata for the benchmark model."""
    from benchmarks.hub_model import get_model_descriptor

    return get_model_descriptor(model_id, cache_dir)


@pytest.fixture(scope="session")
def safetensors_files(model_descriptor):
    """List of all safetensors shard paths for the model."""
    return model_descriptor.safetensors_files


@pytest.fixture(scope="session")
def safetensors_dir(model_descriptor):
    """Directory containing the model's SafeTensors shards."""
    return str(model_descriptor.safetensors_files[0].parent)


@pytest.fixture(scope="session")
def safetensors_path(model_descriptor):
    """Path to the first (or only) safetensors file."""
    return str(model_descriptor.safetensors_files[0])


@pytest.fixture(scope="session")
def serverlessllm_dir(model_id, cache_dir):
    """Path to a ServerlessLLM artifact for the model.

    Uses the benchmark's explicit partition count and supports multi-shard models.
    """
    from benchmarks.hub_model import get_or_build_serverlessllm

    sllm_dir, _ = get_or_build_serverlessllm(model_id, cache_dir)
    return str(sllm_dir)
