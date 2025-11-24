#!/usr/bin/env python3
"""
Download and prepare model fixtures for tensor_store benchmarks.

Downloads SafeTensors format models from HuggingFace and prepares them
for benchmarking with both SafeTensors and ServerlessLLM formats.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, List
import json
import hashlib
from dataclasses import dataclass

try:
    from huggingface_hub import snapshot_download, HfApi, hf_hub_download
    import requests
    from tqdm import tqdm
    import safetensors
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: uv add huggingface-hub requests tqdm safetensors")
    sys.exit(1)


@dataclass
class ModelConfig:
    repo_id: str
    category: str
    description: str
    expected_size_gb: float
    alternative_repo: Optional[str] = None


# Model configurations for different categories
MODEL_CONFIGS = {
    "qwen2-0.5b": ModelConfig(
        repo_id="Qwen/Qwen2-0.5B",
        category="qwen2-0.5b",
        description="Qwen2 0.5B parameter model - efficient small LLM",
        expected_size_gb=1.0,
        alternative_repo="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ),
    "phi2-2.7b": ModelConfig(
        repo_id="microsoft/phi-2",
        category="phi2-2.7b",
        description="Microsoft Phi-2 2.7B parameter model - state-of-the-art small model",
        expected_size_gb=5.5,
        alternative_repo="stabilityai/stablelm-2-1_6b",
    ),
    "mistral-7b": ModelConfig(
        repo_id="mistralai/Mistral-7B-v0.1",
        category="mistral-7b",
        description="Mistral 7B v0.1 - industry standard 7B parameter model",
        expected_size_gb=14.0,
        alternative_repo="tiiuae/falcon-7b",
    ),
    "qwen3-14b": ModelConfig(
        repo_id="Qwen/Qwen3-14B",
        category="qwen3-14b",
        description="Qwen3 14B parameter model - large language model",
        expected_size_gb=28.0,
        alternative_repo="meta-llama/Llama-2-13b-hf",
    ),
    "yi-20b": ModelConfig(
        repo_id="01-ai/Yi-1.5-20B",
        category="yi-20b",
        description="Yi 1.5 20B parameter model - high-performance large model",
        expected_size_gb=40.0,
        alternative_repo="mistralai/Mixtral-8x7B-v0.1",
    ),
}


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


def create_model_readme(
    output_dir: str, config: ModelConfig, safetensors_files: List[str], model_info: dict
):
    """Create README.md with model information."""
    readme_path = Path(output_dir) / "README.md"

    total_size = sum(os.path.getsize(f) for f in safetensors_files)
    total_size_gb = total_size / (1024**3)

    content = f"""# {config.category.title()} Model Fixture

## Model Information
- **Repository**: {config.repo_id}
- **Description**: {config.description}
- **Category**: {config.category}
- **Expected Size**: ~{config.expected_size_gb:.1f} GB
- **Actual Size**: {total_size_gb:.2f} GB
- **Files**: {len(safetensors_files)} SafeTensors file(s)

## HuggingFace Metadata
- **Author**: {model_info.get('author', 'Unknown')}
- **Downloads**: {model_info.get('downloads', 'Unknown')}
- **Likes**: {model_info.get('likes', 'Unknown')}
- **Created**: {model_info.get('created_at', 'Unknown')}

## Files
"""

    for i, filepath in enumerate(safetensors_files):
        size_mb = os.path.getsize(filepath) / (1024**2)
        hash_val = get_file_hash(filepath)[:16]  # Short hash for display
        content += f"- {Path(filepath).name} ({size_mb:.1f} MB, hash: {hash_val})\n"

    content += "\n## Usage\n"
    content += "This fixture contains both SafeTensors and ServerlessLLM formats for benchmarking.\n"
    content += "- `model.safetensors`: Original SafeTensors format\n"
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
        "--category",
        choices=["qwen2-0.5b", "phi2-2.7b", "mistral-7b", "qwen3-14b", "yi-20b"],
        help="Use a predefined model category instead of specifying --repo",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all predefined categories"
    )
    parser.add_argument(
        "--output-dir",
        default="../fixtures",
        help="Output directory relative to scripts/",
    )
    parser.add_argument("--token", help="HuggingFace token for private repos")
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Automatically convert to ServerlessLLM format after download",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=8,
        help="Number of partitions for ServerlessLLM conversion",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify SafeTensors file integrity after download",
    )

    args = parser.parse_args()

    if not args.repo and not args.category and not args.all:
        parser.error(
            "Must specify a HuggingFace repo ID (e.g., 'Qwen/Qwen2-0.5B'), --category, or --all"
        )

    # Determine which models to download
    models_to_download = []

    if args.all:
        models_to_download = list(MODEL_CONFIGS.values())
    elif args.repo:
        # Any HuggingFace model by repo ID
        # Derive a directory name from the repo ID (e.g., "Qwen/Qwen2-0.5B" -> "qwen2-0.5b")
        repo_name = args.repo.split("/")[-1].lower()
        config = ModelConfig(
            repo_id=args.repo,
            category=repo_name,
            description=f"Model: {args.repo}",
            expected_size_gb=1.0,
        )
        models_to_download = [config]
    elif args.category:
        if args.category not in MODEL_CONFIGS:
            parser.error(f"Unknown category: {args.category}")
        models_to_download = [MODEL_CONFIGS[args.category]]

    # Download each model
    for config in models_to_download:
        print(f"\n{'='*60}")
        print(f"Processing {config.category}: {config.repo_id}")
        print(f"{'='*60}")

        output_dir = Path(args.output_dir) / config.category
        temp_dir = output_dir / "tmp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get model info
            model_info = get_model_info(config.repo_id)

            # Download SafeTensors files
            safetensors_files = download_safetensors_model(
                config.repo_id, str(temp_dir), args.token
            )

            # Verify files if requested
            if args.verify:
                print("Verifying SafeTensors files...")
                for filepath in safetensors_files:
                    if not verify_safetensors_file(filepath):
                        print(f"Warning: Verification failed for {filepath}")

            # Create README
            create_model_readme(str(output_dir), config, safetensors_files, model_info)

            # Handle sharded models - use the largest file for fixtures
            if len(safetensors_files) > 1:
                # For sharded models, use the largest shard as the "main" model
                main_file = max(safetensors_files, key=os.path.getsize)
                print(f"Using largest shard for fixture: {Path(main_file).name}")
            else:
                main_file = safetensors_files[0]

            # Copy to final location
            final_safetensors = output_dir / "model.safetensors"
            shutil.copy2(main_file, final_safetensors)
            print(f"Copied to: {final_safetensors}")

            # Convert to ServerlessLLM if requested
            if args.convert:
                convert_binary = Path("../target/release/convert")
                if not convert_binary.exists():
                    print("Warning: convert binary not found, skipping conversion")
                    print("Build with: cargo build --release --bin convert")
                else:
                    serverlessllm_dir = output_dir / "model_serverlessllm"
                    print(
                        f"Converting to ServerlessLLM format with {args.partitions} partitions..."
                    )

                    import subprocess

                    result = subprocess.run(
                        [
                            str(convert_binary),
                            str(final_safetensors),
                            str(serverlessllm_dir),
                            str(args.partitions),
                        ],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        print("✓ Conversion completed successfully!")
                    else:
                        print(f"✗ Conversion failed: {result.stderr}")

            # Clean up temp directory
            shutil.rmtree(temp_dir)
            print(f"✓ Completed {config.category} fixture setup")

        except Exception as e:
            print(f"✗ Failed to process {config.category}: {e}")
            # Clean up on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            continue


if __name__ == "__main__":
    main()

