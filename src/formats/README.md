# Tensor Formats

Checkpoint format parsers and serializers implementing a shared `Model` trait.

## Modules

| Module | Description |
|--------|-------------|
| [`safetensors/`](safetensors/) | HuggingFace SafeTensors format (single/multi-shard) |
| [`serverlessllm/`](serverlessllm/) | ServerlessLLM partitioned layout |
| `traits.rs` | Common interfaces — `Model`, `TensorView`, `SyncSerializer`, `AsyncSerializer` |
| `error.rs` | Unified error types (`ReaderError`, `WriterError`) |

## Usage

```rust
use tensora::formats::safetensors::Model;
use tensora::formats::serverlessllm::Model as SllmModel;
use tensora::formats::traits::Model as ModelTrait;
```

Both format modules implement the `Model` trait, enabling generic code over checkpoint formats.
