# SafeTensors Format

Whole-file and multi-shard eager loading for HuggingFace SafeTensors checkpoints with adaptive storage-engine selection.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Public API re-exports |
| `model.rs` | Model loading (sync, async, io_uring, mmap) |
| `serializer.rs` | SafeTensors serialization |

## API

```rust
use tensora::safetensors::{MmapModel, Model};

// Adaptive loading (picks best I/O backend automatically)
let model = Model::load("model_dir").await?;

// Explicit I/O backend
let model = Model::load_sync("model_dir")?;
let model = Model::load_async("model_dir").await?;
let model = Model::load_io_uring("model_dir")?;  // Linux only
let model = MmapModel::open("model_dir")?;
```

## Storage-Engine Selection

- Single-shard or small multi-shard → `sync`
- Large multi-shard (≥ 4 GB) → `io_uring`
- Non-Linux → `sync` or `async`

## Testing

```bash
cargo test safetensors
cargo bench --bench safetensors
```
