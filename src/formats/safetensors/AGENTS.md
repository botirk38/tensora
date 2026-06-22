# AGENTS.md — SafeTensors

## Module Scope

HuggingFace SafeTensors format: single-file and multi-source-file eager loading.

## Files

- `model.rs` — Loading logic for explicitly selected backends
- `checkpoint.rs` — `Checkpoint` type (in-memory model with serialization)
- `tensor.rs` — `Tensor` and `TensorEntry` view types
- `serializer.rs` — Write SafeTensors files
- `mod.rs` — Public re-exports

## Conventions

- Source files discovered by glob: `*-of-*.safetensors` or single `*.safetensors`
- Header parsed per the SafeTensors spec (8-byte length prefix + JSON)
- Data accessed as contiguous byte slices per tensor
- "Shard" terminology removed in favour of "file" / "source file"
