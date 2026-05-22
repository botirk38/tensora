# AGENTS.md — SafeTensors

## Module Scope

HuggingFace SafeTensors format: single-file and multi-shard eager loading.

## Files

- `model.rs` — All loading logic + backend selection heuristics
- `serializer.rs` — Write SafeTensors files
- `mod.rs` — Public re-exports

## Heuristics (in `model.rs`)

- Single/small-shard → `sync` (thread-parallel chunked reads)
- Large multi-shard (≥ 4 GB) → `io_uring` (multi-worker)
- Non-Linux fallback → `sync`

## Conventions

- Shards discovered by glob: `*-of-*.safetensors` or single `*.safetensors`
- Header parsed per the SafeTensors spec (8-byte length prefix + JSON)
- Data accessed as contiguous byte slices per tensor
