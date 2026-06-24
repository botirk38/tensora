"""Modal infrastructure: app, images, volumes, and compute configuration.

All Modal resource objects live here. Other modules import from here
rather than constructing Modal primitives themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import modal

# ── Build configuration ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Immutable build parameters for Modal container images."""

    rust_version: str = "1.92.0"
    repo_url: str = "https://github.com/botirk38/tensora.git"
    repo_branch: str = "devin/1780968167-paper-revamp-modal"
    git_commit: str = "81e577a"
    workspace: str = "/workspace/tensora"

    @property
    def profile_bin(self) -> str:
        return f"{self.workspace}/target/release/profile"


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    """Immutable compute resource constants for Modal containers."""

    gpu: str = "H100!"
    ephemeral_disk_mib: int = 524_288  # 512 GiB
    memory_mib: int = 32_768  # 32 GiB
    timeout_s: int = 3600  # 1 h max per container
    subprocess_timeout_s: int = 7200  # 2 h max per subprocess
    max_retries: int = 2
    backoff_coefficient: float = 2.0
    initial_delay_s: float = 5.0


BUILD = BuildConfig()
COMPUTE = ComputeConfig()

# ── App singleton ─────────────────────────────────────────────────────

app = modal.App()

# ── Persistent volume: HuggingFace model cache ────────────────────────

hf_volume = modal.Volume.from_name("tensora-hf-cache", create_if_missing=True)
HF_CACHE_MOUNT = "/root/.cache/huggingface"

# ── Rust profiler image ───────────────────────────────────────────────

_PATH_ENV = "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_APT_PACKAGES = [
    "build-essential",
    "pkg-config",
    "libssl-dev",
    "git",
    "curl",
    "ca-certificates",
]

rust_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*_APT_PACKAGES)
    .run_commands(
        f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        f"| sh -s -- -y --default-toolchain {BUILD.rust_version}",
    )
    .env({"PATH": _PATH_ENV, "GIT_COMMIT": BUILD.git_commit})
    .run_commands(
        f"git clone --depth 1 --branch {BUILD.repo_branch} {BUILD.repo_url} {BUILD.workspace}",
        f"cd {BUILD.workspace} && cargo build --release --bin profile",
    )
)

# ── vLLM profiler image ──────────────────────────────────────────────

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*_APT_PACKAGES)
    .run_commands(
        f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        f"| sh -s -- -y --default-toolchain {BUILD.rust_version}",
    )
    .env(
        {
            "PATH": _PATH_ENV,
            "HF_HUB_DISABLE_XET": "1",
            "GIT_COMMIT": BUILD.git_commit,
        }
    )
    .run_commands(
        f"git clone --depth 1 --branch {BUILD.repo_branch} {BUILD.repo_url} {BUILD.workspace}",
    )
    .pip_install("uv")
    .run_commands(
        f"cd {BUILD.workspace}/bindings/python && "
        "uv sync --group torch && "
        "uv run maturin develop --release",
        f"cd {BUILD.workspace}/bindings/python && uv pip install 'vllm>=0.8,<1.0'",
    )
)
