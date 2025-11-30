# Converters Module

This module provides **high-level conversion orchestration** between different checkpoint formats.

## Purpose

Converters are responsible for:
- **Orchestrating** the conversion process from input to output formats
- **Implementing conversion logic** (dtype mapping, shape transformations, partitioning)
- **Coordinating** readers and writers
- **Managing** the overall conversion workflow

## Architecture Principle

**Converters contain ALL conversion logic.** They:
- Use readers to parse input formats
- Apply transformations and mappings
- Use writers to serialize output formats
- Handle all format-specific conversion details

## Module Structure

```
converters/
├── mod.rs                              # Module exports
├── safetensors_to_serverlessllm.rs    # SafeTensors → ServerlessLLM conversion
└── safetensors_to_tensorstore.rs       # SafeTensors → TensorStore conversion
```

## Key Functions

### SafeTensors to ServerlessLLM
```rust
use tensor_store::converters::safetensors_to_serverlessllm;

// Convert with partitioning
convert_safetensors_to_serverlessllm(
    "model.safetensors",    // input path
    "output_dir/",          // output directory
    8,                      // partition count
).await?;
```

## Conversion Logic

Converters implement format-specific transformations:

### Dtype Mapping
```rust
// SafeTensors Dtype → ServerlessLLM string
fn dtype_to_serverlessllm_string(dtype: safetensors::Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "torch.float32",
        Dtype::F16 => "torch.float16",
        // ... more mappings
    }
}
```

### Shape and Stride Calculations
```rust
// Calculate contiguous strides from shape
fn calculate_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (1..shape.len()).rev() {
        strides[i-1] = strides[i] * shape[i] as i64;
    }
    strides
}
```

### Partitioning Logic
```rust
// Assign tensors to partitions
fn partition_tensors(tensor_count: usize, partition_count: usize) -> Vec<usize> {
    (0..tensor_count)
        .map(|i| i % partition_count)
        .collect()
}
```

## Usage Pattern

```rust
// 1. Choose appropriate converter for input→output format pair
use tensor_store::converters::safetensors_to_serverlessllm;

// 2. Call converter with input/output paths and parameters
convert_safetensors_to_serverlessllm(
    input_path,
    output_dir,
    partition_count,
).await?;

// 3. Converter handles the full pipeline:
//    - Load input using readers::safetensors
//    - Apply conversions (dtypes, shapes, partitioning)
//    - Write output using writers::serverlessllm
```

## Design Rationale

**Why separate converters from readers/writers?**

1. **Single Responsibility**: Readers parse, writers serialize, converters transform
2. **Reusability**: Same reader can be used by multiple converters
3. **Testability**: Each component can be tested independently
4. **Maintainability**: Conversion logic is centralized and versioned together
5. **Extensibility**: Easy to add new format conversions without touching core I/O

## Future Extensions

Additional converters can be added for:
- `safetensors_to_tensorstore.rs`
- `tensorstore_to_serverlessllm.rs`
- `serverlessllm_to_safetensors.rs`
- etc.