# CLI Binaries

Command-line tools for profiling, demonstrating, and converting LLM checkpoints.

## Binaries

| Binary | Purpose |
|--------|---------|
| [`profile/`](profile/) | Performance measurement harness for all backends |
| [`demo/`](demo/) | Interactive demonstrations of loading strategies |
| [`convert/`](convert/) | SafeTensors → ServerlessLLM format conversion |

## Usage

All binaries accept a HuggingFace model ID via `--model-id`:

```bash
cargo run --release --bin profile -- safetensors default --model-id Qwen/Qwen3-0.6B
cargo run --release --bin demo -- safetensors all --model-id Qwen/Qwen3-0.6B
cargo run --release --bin convert -- ./input ./output
```

Models are resolved via the HuggingFace Hub cache. First run downloads shards automatically.
