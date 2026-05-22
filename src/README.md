# src/

Core Rust library for Tensora.

## Modules

| Module | Description |
|--------|-------------|
| [`backends/`](backends/) | I/O backends — sync, async (Tokio), io_uring, mmap |
| [`formats/`](formats/) | Checkpoint format parsers and serializers |
| [`converters/`](converters/) | Format-to-format conversion pipelines |
| [`hf_model.rs`](hf_model.rs) | HuggingFace Hub model resolution |
| [`bin/`](bin/) | CLI binaries (profile, demo, convert) |
| [`lib.rs`](lib.rs) | Public API and re-exports |

## Entry Point

`lib.rs` re-exports the most commonly used types:

```rust
use tensora::{SafeTensorsModel, ServerlessLLMModel, backends};
```

## Platform Notes

- `io_uring` and `odirect` backends are gated behind `#[cfg(target_os = "linux")]`
- All other modules are cross-platform
