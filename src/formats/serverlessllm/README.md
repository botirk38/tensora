# ServerlessLLM Format

Partitioned tensor loading for tensora, with adaptive backend selection.

## Module Structure

```
serverlessllm/
├── mod.rs          # Public API re-exports
├── model.rs        # Model loading (sync, async, io_uring, mmap)
├── index.rs        # Partition index parsing
├── serializer.rs   # ServerlessLLM serialization
├── helpers.rs      # Partition count heuristics
└── tensor.rs       # Tensor view types
```

## Loading Models

The `Model` type provides adaptive loading through `Model::load(path).await`, which selects the best backend based on workload size and partition structure. Explicit paths are also available:

- `Model::load_sync(path)` — blocking POSIX reads
- `Model::load_async(path).await` — Tokio-based async loading with range batching
- `Model::load_io_uring(path)` — Linux io_uring with multi-worker submission
- `Model::load_mmap(path)` — memory-mapped lazy access

### Backend Selection

The `default` policy uses workload-aware heuristics:

- Smaller range-heavy workloads → `async`
- Larger partitioned workloads → `io_uring`
- Non-Linux platforms → `async`

## API

```rust
use tensora::serverlessllm::Model;

let model = Model::load("model_serverlessllm").await?;
for name in model.tensor_names() {
    let tensor = model.tensor(&name)?;
    println!("{}: shape={:?}, dtype={:?}", name, tensor.shape(), tensor.dtype());
}
```

## Partitioned Layout

ServerlessLLM stores tensors across partition files for range-oriented access:

```
model_serverlessllm/
├── metadata.json    # Partition metadata
├── tensor.data_0    # Partition 0
├── tensor.data_1    # Partition 1
└── ...
```

The loader automatically discovers partitions and loads tensors through coalesced range requests.

## Testing

```bash
cargo test serverlessllm
cargo bench --bench serverlessllm
```

## References

- [ServerlessLLM Paper (OSDI '24)](https://www.usenix.org/conference/osdi24/presentation/fu)
- [ServerlessLLM GitHub](https://github.com/ServerlessLLM/ServerlessLLM)
