# Converters

High-level format conversion pipelines between checkpoint layouts.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Module exports |
| `safetensors_to_serverlessllm.rs` | SafeTensors → ServerlessLLM conversion with partitioning |

## API

```rust
use tensora::convert_safetensors_to_serverlessllm;

// Adaptive (picks best I/O backend)
convert_safetensors_to_serverlessllm("input_dir", "output_dir", 8).await?;

// Explicit backends
convert_safetensors_to_serverlessllm_sync("input_dir", "output_dir", 8)?;
convert_safetensors_to_serverlessllm_async("input_dir", "output_dir", 8).await?;
```

## Design

Converters orchestrate the full pipeline:
1. Parse input format via format readers
2. Apply transformations (dtype mapping, stride calculation, partitioning)
3. Write output via format serializers

This separation keeps readers/writers reusable and conversion logic centralized.

## Testing

```bash
cargo test converters
cargo bench --bench conversion
```
