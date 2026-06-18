# AGENTS.md — Formats

## Responsibility

Parse and serialize checkpoint formats. Each format module provides a `Model` type implementing the `Model` trait from `traits.rs`.

## Key Traits

- `Model` — tensor access (`tensor_names()`, `tensor()`)
- `TensorView` — tensor metadata (`shape()`, `dtype()`, `data()`)
- `SyncSerializer` / `AsyncSerializer` — serialization interfaces

## Conventions

- Storage-engine selection heuristics live in each format's `model.rs`
- Error types are unified in `error.rs` (`LoadError`, `SaveError`)
- Multi-shard discovery is automatic (glob for shard patterns)
- `MmapModel` variants provide lazy loading without full materialization

## Adding a New Format

1. Create `src/formats/<name>/` with `mod.rs`, `model.rs`, (optionally) `serializer.rs`
2. Implement `Model` and `TensorView` traits
3. Re-export in `src/formats/mod.rs`
4. Add convenience re-exports in `src/lib.rs`

## Do NOT

- Put I/O strategy logic in serializers (that belongs in `model.rs`)
- Return raw bytes without a `TensorView` wrapper
