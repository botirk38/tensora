# ServerlessLLM Format

Partitioned tensor loading for range-oriented access patterns with adaptive storage-engine selection.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Public API re-exports |
| `model.rs` | Model loading (sync, async, io_uring, mmap) |
| `index.rs` | Partition index (`metadata.json`) parsing |
| `serializer.rs` | ServerlessLLM serialization |
| `helpers.rs` | Partition count heuristics |
| `tensor.rs` | Tensor view types and `TensorMmap` |

## Layout

```
model_serverlessllm/
├── metadata.json    # Tensor index with partition assignments
├── tensor.data_0    # Partition 0
├── tensor.data_1    # Partition 1
└── ...
```

## API

```rust
use tensora::serverlessllm::Model;

let model = Model::load("model_serverlessllm").await?;
for name in model.tensor_names() {
    let t = model.tensor(&name)?;
    println!("{}: {:?}", name, t.shape());
}
```

## Storage-Engine Selection

- Range-heavy workloads → `async`
- Large partitioned workloads → `io_uring`
- Non-Linux → `async`

## Testing

```bash
cargo test serverlessllm
cargo bench --bench serverlessllm
```

## References

- [ServerlessLLM (OSDI '24)](https://www.usenix.org/conference/osdi24/presentation/fu)
