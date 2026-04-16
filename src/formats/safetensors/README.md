# SafeTensors Format

SafeTensors support for tensora: whole-file and multi-shard eager loading with adaptive backend selection.

## Module Structure

```
safetensors/
├── mod.rs          # Public API re-exports
├── model.rs        # Model loading (sync, async, io_uring, mmap)
└── serializer.rs   # SafeTensors serialization
```

## Loading Models

The `Model` type provides adaptive loading through `Model::load(path).await`, which selects the best backend based on checkpoint size and shard structure. Explicit paths are also available:

- `Model::load_sync(path)` — blocking POSIX reads with dynamic chunking
- `Model::load_async(path).await` — Tokio-based async loading
- `Model::load_io_uring(path)` — Linux io_uring with multi-worker submission
- `Model::load_mmap(path)` — memory-mapped lazy access

### Backend Selection

The `default` policy uses workload-aware heuristics:

- Single-shard or small multi-shard checkpoints → `sync`
- Large multi-shard checkpoints (≥ ~4 GB total) → `io_uring`
- Non-Linux platforms → `sync` or `async`

## API

```rust
use tensora::safetensors::Model;

let model = Model::load("model_dir").await?;
for name in model.tensor_names() {
    let view = model.tensor(&name)?;
    println!("{}: shape={:?}, dtype={:?}", name, view.shape(), view.dtype());
}
```

## Multi-Shard Support

SafeTensors models often span multiple shard files (e.g., `model-00001-of-00004.safetensors`). The loader automatically discovers shards in a directory and loads them as a unified model.

## Testing

```bash
cargo test safetensors
cargo bench --bench safetensors
```

## References

- [SafeTensors Format Specification](https://github.com/huggingface/safetensors)
- [HuggingFace SafeTensors Library](https://github.com/huggingface/safetensors)
