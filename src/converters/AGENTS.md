# AGENTS.md — Converters

## Responsibility

Orchestrate format-to-format conversion. Converters use format readers and writers but own all transformation logic.

## Design Principle

Converters contain ALL conversion logic:
- Dtype mapping (SafeTensors enum → ServerlessLLM string)
- Shape/stride calculation
- Partition assignment
- I/O coordination

## Key Types

- `ConversionPlan` — describes the full conversion before execution
- `ConversionStats` — timing and throughput metrics
- `CopyOp` / `TensorSource` — individual copy operations

## Adding a New Converter

1. Create `src/converters/<from>_to_<to>.rs`
2. Implement sync, async, and (optionally) io_uring variants
3. Export from `mod.rs`
4. Add re-exports in `lib.rs`

## Do NOT

- Put conversion logic inside format modules
- Skip the planning phase (always build a `ConversionPlan` first)
