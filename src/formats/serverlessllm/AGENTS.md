# AGENTS.md — ServerlessLLM

## Module Scope

ServerlessLLM partitioned tensor format: range-oriented loading for cold-start optimization.

## Files

- `model.rs` — Loading logic + storage-engine selection heuristics
- `index.rs` — `metadata.json` parsing (partition assignments, tensor offsets)
- `serializer.rs` — Write ServerlessLLM layouts
- `helpers.rs` — Partition count recommendations
- `tensor.rs` — `Tensor` and `TensorMmap` view types
- `mod.rs` — Public re-exports

## Heuristics (in `model.rs`)

- Range-heavy workloads → `async` (Tokio grouped tasks)
- Large partitioned → `io_uring`
- Non-Linux → `async`

## Layout

```
metadata.json + tensor.data_0 .. tensor.data_{N-1}
```

## Conventions

- Partition count default: `max(1, ceil(bytes / 512 MiB))`
- Tensors can span partition boundaries (coalesced reads)
- `TensorMmap` provides lazy access without copying
