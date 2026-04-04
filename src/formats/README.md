# Tensor Formats

## Modules

- **safetensors/** — SafeTensors format reading/writing
- **serverlessllm/** — ServerlessLLM format (partitioned tensor layout)
- **traits.rs** — Common interfaces (Model, TensorView, Serializers)
- **error.rs** — Error types

## Usage

```rust
use tensor_store::safetensors;
use tensor_store::serverlessllm;
```

Each format module exposes `Model` and `TensorView` implementations via the traits in `traits.rs`. Choose based on your source checkpoint format:

- **SafeTensors**: HuggingFace standard, single or multi-shard
- **ServerlessLLM**: Partitioned layout for range-oriented access, requires conversion