# SafeTensors Format

Whole-file and multi-shard eager loading for HuggingFace SafeTensors checkpoints with adaptive backend selection.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Public API re-exports |
| `model.rs` | Model loading (sync, async, io_uring, mmap) |
| `serializer.rs` | SafeTensors serialization |

## API

```rust
use tensora::safetensors::Model;

// Adaptive loading (picks best backend automatically)
let model = Model::load("model_dir").await?;

// Explicit backend
let model = Model::load_sync("model_dir")?;
let model = Model::load_async("model_dir").await?;
let model = Model::load_io_uring("model_dir")?;  // Linux only
let model = Model::load_mmap("model_dir")?;
```

## Backend Selection

- Single-shard or small multi-shard → `sync`
- Large multi-shard (≥ 4 GB) → `io_uring`
- Non-Linux → `sync` or `async`

## Testing

```bash
cargo test safetensors
cargo bench --bench safetensors
```
