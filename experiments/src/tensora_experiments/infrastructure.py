"""Modal infrastructure declarations: app, image, volumes, and resource constants.

This module owns all Modal resource objects. Other modules import from here
rather than constructing Modal primitives themselves.
"""

import modal

# ---------------------------------------------------------------------------
# App singleton
# ---------------------------------------------------------------------------

app = modal.App()

# ---------------------------------------------------------------------------
# Container image: Debian + Rust 1.92 + tensora profile binary
# ---------------------------------------------------------------------------

RUST_VERSION = "1.92.0"
REPO_URL = "https://github.com/botirk38/tensora.git"
REPO_BRANCH = "devin/1780968167-paper-revamp-modal"
GIT_COMMIT = "7da8936"
WORKSPACE = "/workspace/tensora"
PROFILE_BIN = f"{WORKSPACE}/target/release/profile"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "git",
        "curl",
        "ca-certificates",
    )
    .run_commands(
        f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        f"| sh -s -- -y --default-toolchain {RUST_VERSION}",
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "GIT_COMMIT": GIT_COMMIT,
        }
    )
    .run_commands(
        f"git clone --depth 1 --branch {REPO_BRANCH} {REPO_URL} {WORKSPACE}",
        f"cd {WORKSPACE} && cargo build --release --bin profile",
    )
)

# ---------------------------------------------------------------------------
# Persistent volume: HuggingFace model weight cache
# ---------------------------------------------------------------------------

hf_volume = modal.Volume.from_name("tensora-hf-cache", create_if_missing=True)
HF_CACHE_MOUNT = "/root/.cache/huggingface"

# ---------------------------------------------------------------------------
# Compute resource constants
# H100! disables auto-upgrade to H200 for benchmark reproducibility.
# ---------------------------------------------------------------------------

GPU = "H100!"
EPHEMERAL_DISK_MIB = 524_288  # 512 GiB — minimum for H100 on Modal
MEMORY_MIB = 32_768  # 32 GiB system RAM
TIMEOUT_S = 3600  # 1 hour max per container lifetime
