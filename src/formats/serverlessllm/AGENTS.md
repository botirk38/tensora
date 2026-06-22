# AGENTS.md — ServerlessLLM

## Module Scope

ServerlessLLM partitioned tensor format: range-oriented loading for cold-start optimization.

## Files

- `model.rs` — Loading logic for explicitly selected backends
- `index.rs` — `metadata.json` parsing (partition assignments, tensor offsets)
- `serializer.rs` — Write ServerlessLLM layouts
- `tensor.rs` — `Tensor` and `TensorMmap` view types
- `mod.rs` — Public re-exports

## Layout

```
metadata.json + tensor.data_0 .. tensor.data_{N-1}
```

## Conventions

- Partition count is caller-provided
- Tensors can span partition boundaries (coalesced reads)
- `TensorMmap` provides lazy access without copying
