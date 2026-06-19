# AGENTS.md — SafeTensors

## Module Scope

HuggingFace SafeTensors format: single-file and multi-source-file eager loading.

## Files

- `model.rs` — All loading logic + storage-engine selection heuristics
- `checkpoint.rs` — `Checkpoint` type (in-memory model with serialization)
- `tensor.rs` — `Tensor` and `TensorEntry` view types
- `serializer.rs` — Write SafeTensors files
- `mod.rs` — Public re-exports

## Heuristics (in `model.rs`)

- Single/small file → `sync` (thread-parallel chunked reads)
- Large multi-file (≥ 4 GB) → `io_uring` (multi-worker)
- Non-Linux fallback → `sync`

## Conventions

- Source files discovered by glob: `*-of-*.safetensors` or single `*.safetensors`
- Header parsed per the SafeTensors spec (8-byte length prefix + JSON)
- Data accessed as contiguous byte slices per tensor
- "Shard" terminology removed in favour of "file" / "source file"
