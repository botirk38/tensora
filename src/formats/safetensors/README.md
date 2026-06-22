# SafeTensors Format

Whole-file and multi-shard eager loading for HuggingFace SafeTensors checkpoints with explicit storage-engine selection.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Public API re-exports |
| `model.rs` | Model loading (sync, async, io_uring, mmap) |
| `serializer.rs` | SafeTensors serialization |

## API

```rust
use tensora::safetensors::{MmapModel, Model};

// Default loading path
let model = Model::load("model_dir").await?;

// Explicit I/O backend
let model = Model::load_sync("model_dir")?;
let model = Model::load_async("model_dir").await?;
let model = Model::load_io_uring("model_dir")?;  // Linux only
let model = MmapModel::open("model_dir")?;
```

## Storage Engines

- `sync` uses blocking positioned I/O.
- `async` uses Tokio tasks.
- `io_uring` is Linux-only.
- `mmap` provides lazy tensor access through `MmapModel`.

## Testing

```bash
cargo test safetensors
cargo bench --bench safetensors
```
