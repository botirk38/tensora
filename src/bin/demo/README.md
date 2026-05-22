# demo

Interactive demonstration of SafeTensors and ServerlessLLM loader capabilities.

## Usage

```bash
cargo run --release --bin demo -- <FORMAT> <SCENARIO> --model-id <HF/MODEL>
```

**Formats:** `safetensors`, `serverlessllm`
**Scenarios:** `async`, `sync`, `mmap`, `metadata`, `all`

## Examples

```bash
cargo run --release --bin demo -- safetensors async --model-id Qwen/Qwen3-0.6B
cargo run --release --bin demo -- serverlessllm all --model-id Qwen/Qwen3-0.6B
```

## Files

| File | Purpose |
|------|---------|
| `main.rs` | CLI entry point |
| `config.rs` | Demo configuration |
| `io_metrics.rs` | I/O metrics collection and display |
| `safetensors.rs` | SafeTensors demo scenarios |
| `serverlessllm.rs` | ServerlessLLM demo scenarios |
