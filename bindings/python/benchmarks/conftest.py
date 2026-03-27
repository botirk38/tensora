"""Pytest configuration for benchmarks: CLI options and fixtures."""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--model-id",
        action="store",
        default="gpt2",
        help="HuggingFace model ID (e.g., gpt2, Qwen/Qwen2-0.5B)",
    )


@pytest.fixture(scope="session")
def model_id(request):
    """Model ID from command line."""
    return request.config.getoption("--model-id")


@pytest.fixture(scope="session")
def safetensors_path(tmp_path_factory):
    """Path to a synthetic SafeTensors file for non-vLLM benchmarks."""
    from benchmarks.fixtures import create_gpt2, write_safetensors

    tmp = tmp_path_factory.mktemp("bench_synthetic")
    tensors = create_gpt2(n_layers=2)
    return str(write_safetensors(tensors, tmp / "model.safetensors"))


@pytest.fixture(scope="session")
def serverlessllm_dir(tmp_path_factory):
    """Session-scoped path to a synthetic ServerlessLLM directory (GPT-2-like)."""
    from benchmarks.fixtures import create_gpt2, write_serverlessllm_dir

    tmp = tmp_path_factory.mktemp("bench_synthetic")
    tensors = create_gpt2(n_layers=2)
    return str(write_serverlessllm_dir(tensors, tmp / "model_sllm"))


@pytest.fixture(scope="session")
def serverlessllm_dir_4partitions(tmp_path_factory):
    """Session-scoped path to a synthetic ServerlessLLM directory with 4 partitions."""
    from benchmarks.fixtures import create_gpt2, write_serverlessllm_dir

    tmp = tmp_path_factory.mktemp("bench_synthetic")
    tensors = create_gpt2(n_layers=2)
    return str(
        write_serverlessllm_dir(tensors, tmp / "model_sllm_4p", num_partitions=4)
    )


@pytest.fixture(scope="session")
def serverlessllm_dir_8partitions(tmp_path_factory):
    """Session-scoped path to a synthetic ServerlessLLM directory with 8 partitions."""
    from benchmarks.fixtures import create_gpt2, write_serverlessllm_dir

    tmp = tmp_path_factory.mktemp("bench_synthetic")
    tensors = create_gpt2(n_layers=2)
    return str(
        write_serverlessllm_dir(tensors, tmp / "model_sllm_8p", num_partitions=8)
    )
