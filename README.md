# tensor_store

Adaptive checkpoint loading for LLMs. Supports sync, async, and io_uring backends. Formats: SafeTensors, ServerlessLLM.

## Quick Start

```bash
cd scripts
uv sync
uv run python download_models.py Qwen/Qwen3-0.6B
cd ..
```

Run demos:

```bash
cargo run --release --bin demo -- safetensors all
cargo run --release --bin demo -- serverlessllm all
```

## Python

```bash
cd bindings/python
uv sync --group dev --group torch
uv run python examples/pytorch.py gpt2 --prompt "Hello"
```

## Tests

```bash
cargo test --lib --locked
cargo clippy --lib --locked -- -D warnings
```

## Docs

- [I/O Backends](src/backends/README.md)
- [SafeTensors](src/formats/safetensors/README.md)
- [ServerlessLLM](src/formats/serverlessllm/README.md)
- [Python Bindings](bindings/python/README.md)

## License

Apache 2.0