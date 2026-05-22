# profile

Performance measurement harness for Tensora I/O backends.

## Usage

```bash
cargo run --release --bin profile -- <FORMAT> <BACKEND> --model-id <HF/MODEL> [--iterations N]
```

**Formats:** `safetensors`, `serverlessllm`
**Backends:** `default`, `sync`, `async`, `mmap`, `io-uring` (Linux only)

## Examples

```bash
cargo run --release --bin profile -- safetensors default --model-id openai-community/gpt2
cargo run --release --bin profile -- serverlessllm sync --model-id Qwen/Qwen3-0.6B -i 3
```

## Cold-Cache Measurements

```bash
sync && echo 3 > /proc/sys/vm/drop_caches
cargo run --release --bin profile -- safetensors sync --model-id Qwen/Qwen3-8B -i 1
```

## Files

| File | Purpose |
|------|---------|
| `main.rs` | CLI entry point and argument parsing |
| `config.rs` | Run configuration |
| `evict.rs` | Cache eviction utilities |
| `safetensors.rs` | SafeTensors profiling logic |
| `serverlessllm.rs` | ServerlessLLM profiling logic |
| `stats.rs` | Timing statistics and reporting |
