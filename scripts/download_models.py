"""
Download and prepare model fixtures for tensor_store benchmarks.

Downloads SafeTensors format models from HuggingFace and prepares them
for benchmarking with both SafeTensors and ServerlessLLM formats.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, List
import hashlib
from dataclasses import dataclass
from huggingface_hub import snapshot_download, HfApi


@dataclass
class ModelConfig:
    repo_id: str
    description: str
    expected_size_gb: float


def repo_dir_name(repo_id: str) -> str:
    """Convert a HuggingFace repo id into a filesystem-friendly directory name."""
    return repo_id.replace("/", "-").lower()


def get_model_info(repo_id: str) -> dict:
    """Get model information from HuggingFace API."""
    try:
        api = HfApi()
        model_info = api.model_info(repo_id)
        return {
            "repo_id": repo_id,
            "model_id": model_info.id,
            "author": model_info.author,
            "downloads": model_info.downloads,
            "likes": model_info.likes,
            "tags": model_info.tags,
            "created_at": (
                model_info.created_at.isoformat() if model_info.created_at else None
            ),
        }
    except Exception as e:
        print(f"Warning: Could not fetch model info for {repo_id}: {e}")
        return {"repo_id": repo_id}


def download_safetensors_model(
    repo_id: str, output_dir: str, token: Optional[str] = None, max_retries: int = 3
) -> List[str]:
    """Download SafeTensors model from HuggingFace."""
    print(f"Downloading {repo_id}...")

    for attempt in range(max_retries):
        try:
            # Download only SafeTensors files
            download_dir = snapshot_download(
                repo_id=repo_id,
                local_dir=output_dir,
                token=token,
                allow_patterns=["*.safetensors"],
                ignore_patterns=[
                    "*.bin",
                    "*.msgpack",
                    "*.h5",
                    "*.pb",
                    "*.onnx",
                    "*.tflite",
                ],
            )

            # Find all downloaded SafeTensors files
            safetensors_files = list(Path(download_dir).glob("*.safetensors"))
            if not safetensors_files:
                raise FileNotFoundError(
                    f"No .safetensors files found in {download_dir}"
                )

            print(f"Downloaded {len(safetensors_files)} SafeTensors file(s)")
            return [str(f) for f in sorted(safetensors_files)]

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download attempt {attempt + 1} failed: {e}. Retrying...")
                continue
            else:
                raise e

    # This should never be reached, but for type safety
    raise RuntimeError(f"Failed to download {repo_id} after {max_retries} attempts")


def verify_safetensors_file(filepath: str) -> bool:
    """Verify SafeTensors file integrity."""
    try:
        from safetensors import safe_open

        with safe_open(filepath, framework="numpy", device="cpu") as f:
            # Just try to open and read metadata
            metadata = f.metadata()
            keys = list(f.keys())
            print(
                f"Verified {filepath}: {len(keys)} tensors, metadata: {metadata is not None}"
            )
        return True
    except Exception as e:
        print(f"Verification failed for {filepath}: {e}")
        return False


def get_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Calculate file hash."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


_RECOMMENDED_PARTITION_TARGET_BYTES = 512 * 1024 * 1024


def recommended_partition_count(file_size_bytes: int) -> int:
    """Default ServerlessLLM partition count: ``max(1, ceil(bytes / 512 MiB))`` (uncapped).

    Matches ``tensor_store::formats::serverlessllm::recommended_partition_count`` in Rust.
    """
    if file_size_bytes <= 0:
        return 1
    return max(
        1,
        (file_size_bytes + _RECOMMENDED_PARTITION_TARGET_BYTES - 1)
        // _RECOMMENDED_PARTITION_TARGET_BYTES,
    )


def create_model_readme(
    output_dir: str, config: ModelConfig, safetensors_files: List[str], model_info: dict
):
    """Create README.md with model information."""
    readme_path = Path(output_dir) / "README.md"

    total_size = sum(os.path.getsize(f) for f in safetensors_files)
    total_size_gb = total_size / (1024**3)

    content = f"""# {config.repo_id} Model Fixture

## Model Information
- **Repository**: {config.repo_id}
- **Description**: {config.description}
- **Expected Size**: ~{config.expected_size_gb:.1f} GB
- **Actual Size**: {total_size_gb:.2f} GB
- **Files**: {len(safetensors_files)} SafeTensors file(s)

## HuggingFace Metadata
- **Author**: {model_info.get("author", "Unknown")}
- **Downloads**: {model_info.get("downloads", "Unknown")}
- **Likes**: {model_info.get("likes", "Unknown")}
- **Created**: {model_info.get("created_at", "Unknown")}

## Files
"""

    for i, filepath in enumerate(safetensors_files):
        size_mb = os.path.getsize(filepath) / (1024**2)
        hash_val = get_file_hash(filepath)[:16]  # Short hash for display
        content += f"- {Path(filepath).name} ({size_mb:.1f} MB, hash: {hash_val})\n"

    content += "\n## Usage\n"
    content += "This fixture contains both SafeTensors and ServerlessLLM formats for benchmarking.\n"
    content += "- `*.safetensors`: Downloaded SafeTensors shards\n"
    content += "- `model_serverlessllm/`: Converted ServerlessLLM format with partitioned data\n"

    with open(readme_path, "w") as f:
        f.write(content)

    print(f"Created README.md at {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download model fixtures for tensor_store. Supports downloading any HuggingFace model with SafeTensors format."
    )
    parser.add_argument(
        "repo",
        nargs="?",
        help="HuggingFace repository ID (e.g., 'Qwen/Qwen2-0.5B', 'meta-llama/Llama-2-7b-hf'). This is the only required argument.",
    )

    parser.add_argument(
        "--output-dir",
        default="../fixtures",
        help="Output directory relative to scripts/",
    )
    parser.add_argument("--token", help="HuggingFace token for private repos")
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip conversion to ServerlessLLM format (only download SafeTensors)",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=None,
        help="Number of partitions for ServerlessLLM conversion (default: auto-calculated based on model size)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify SafeTensors file integrity after download",
    )

    args = parser.parse_args()

    if not args.repo:
        parser.error("Must specify a HuggingFace repo ID (e.g., 'Qwen/Qwen2-0.5B')")

    # Determine which models to download (only repo id is required)
    config = ModelConfig(
        repo_id=args.repo,
        description=f"Model: {args.repo}",
        expected_size_gb=1.0,
    )
    models_to_download = [config]

    # Download each model
    for config in models_to_download:
        print(f"\n{'=' * 60}")
        print(f"Processing {config.repo_id}")
        print(f"{'=' * 60}")

        output_dir = Path(args.output_dir) / repo_dir_name(config.repo_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        stale_single = output_dir / "model.safetensors"
        if stale_single.exists():
            stale_single.unlink()
        stale_artifact = output_dir / "model_serverlessllm"
        if stale_artifact.exists():
            shutil.rmtree(stale_artifact)

        try:
            # Get model info
            model_info = get_model_info(config.repo_id)

            # Download SafeTensors files
            safetensors_files = download_safetensors_model(
                config.repo_id, str(output_dir), args.token
            )

            # Verify files if requested
            if args.verify:
                print("Verifying SafeTensors files...")
                for filepath in safetensors_files:
                    if not verify_safetensors_file(filepath):
                        print(f"Warning: Verification failed for {filepath}")

            # Create README
            create_model_readme(str(output_dir), config, safetensors_files, model_info)

            # Convert to ServerlessLLM by default (unless --no-convert specified)
            if not args.no_convert:
                convert_binary = Path("../target/release/convert")

                # Auto-build if missing
                if not convert_binary.exists():
                    print("Convert binary not found. Building...")
                    import subprocess

                    build_result = subprocess.run(
                        ["cargo", "build", "--release", "--bin", "convert"],
                        cwd=Path(".."),
                        capture_output=True,
                        text=True,
                    )

                    if build_result.returncode != 0:
                        print(
                            f"✗ Failed to build convert binary: {build_result.stderr}"
                        )
                        print("Skipping ServerlessLLM conversion")
                    else:
                        print("✓ Convert binary built successfully")

                # Proceed with conversion if binary exists now
                if convert_binary.exists():
                    serverlessllm_dir = output_dir / "model_serverlessllm"
                    if serverlessllm_dir.exists():
                        shutil.rmtree(serverlessllm_dir)

                    # Calculate default partitions if not specified
                    file_size = sum(os.path.getsize(f) for f in safetensors_files)
                    partition_count = (
                        args.partitions
                        if args.partitions is not None
                        else recommended_partition_count(file_size)
                    )

                    print(
                        f"Converting to ServerlessLLM format with {partition_count} partitions (model size: {file_size / (1024**3):.2f} GB)..."
                    )

                    import subprocess

                    result = subprocess.run(
                        [
                            str(convert_binary),
                            str(output_dir),
                            str(serverlessllm_dir),
                            "--partitions",
                            str(partition_count),
                        ],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        print("✓ Conversion completed successfully!")
                    else:
                        print(f"✗ Conversion failed: {result.stderr}")

            print(f"✓ Completed fixture setup for {config.repo_id}")

        except Exception as e:
            print(f"✗ Failed to process {config.repo_id}: {e}")
            continue


if __name__ == "__main__":
    main()
