# SafeTensors Module

Implementation of SafeTensors format reading and writing for tensor_store.

## Overview

SafeTensors is a simple, safe format for storing tensors securely. This module provides:
- High-performance reading with multiple backend support
- Writing tensors to SafeTensors format
- Zero-copy operations where possible

## Format Specification

SafeTensors uses a simple binary format:
```
[8 bytes: header_size (little-endian u64)]
[header_size bytes: JSON metadata]
[remaining bytes: tensor data]
```

The JSON header contains tensor metadata:
```json
{
  "tensor_name": {
    "dtype": "F32",
    "shape": [2, 3],
    "data_offsets": [0, 24]
  }
}
```

## Module Structure

```
safetensors/
├── mod.rs     # Public API re-exports
├── reader.rs  # SafeTensors reading implementations
└── writer.rs  # SafeTensors writing implementations
```

## Reading SafeTensors

### Async Loading

```rust
use tensor_store::safetensors;

// Load entire file
let tensors = safetensors::load("model.safetensors").await?;
println!("Loaded {} tensors", tensors.names().len());

// Access tensor data
let tensor = tensors.tensor("weight").unwrap();
println!("Shape: {:?}", tensor.shape());
println!("Dtype: {:?}", tensor.dtype());
let data: &[f32] = tensor.data();

// Parallel loading for large files
let tensors = safetensors::load_parallel("model.safetensors", 4).await?;
```

### Synchronous Loading

```rust
use tensor_store::safetensors;

// Blocking load (no async runtime needed)
let tensors = safetensors::load_sync("model.safetensors")?;

// Load specific byte range
let chunk = safetensors::load_range_sync("model.safetensors", offset, length)?;
```

### Memory-Mapped Loading

```rust
use tensor_store::safetensors;

// Memory-map for lazy loading
let tensors = safetensors::load_mmap("model.safetensors")?;

// Data is loaded on access
for name in tensors.tensors().names() {
    let tensor = tensors.tensors().tensor(name)?;
    // Page fault occurs here on first access
    let data = tensor.data();
}
```

## Writing SafeTensors

### Basic Writing

```rust
use tensor_store::safetensors::{SafeTensorsWriter, TensorView, Dtype};

let mut writer = SafeTensorsWriter::new();

// Add tensors
let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
writer.add_tensor(
    "model.weight",
    TensorView {
        dtype: Dtype::F32,
        shape: vec![2, 2],
        data: bytemuck::cast_slice(&weights),
    }
)?;

// Write to file
writer.write_to_file("output.safetensors").await?;
```

### With Metadata

```rust
use std::collections::HashMap;

let mut writer = SafeTensorsWriter::new();

// Add custom metadata
let mut metadata = HashMap::new();
metadata.insert("model_type".to_string(), "transformer".to_string());
metadata.insert("version".to_string(), "1.0".to_string());
writer.set_metadata(metadata);

// Add tensors...
writer.add_tensor("weight", tensor_view)?;

// Write
writer.write_to_file("model.safetensors").await?;
```

## Types

### SafeTensorsOwned

Owned SafeTensors data loaded into memory:

```rust
pub struct SafeTensorsOwned {
    // Owned tensor data
}

impl SafeTensorsOwned {
    pub fn names(&self) -> Vec<&str>;
    pub fn tensor(&self, name: &str) -> Option<Tensor>;
    pub fn into_bytes(self) -> Vec<u8>;
}
```

### SafeTensorsMmap

Memory-mapped SafeTensors (zero-copy):

```rust
pub struct SafeTensorsMmap {
    // Memory-mapped file
}

impl SafeTensorsMmap {
    pub fn tensors(&self) -> SafeTensors<'_>;
}
```

### Tensor

Individual tensor view:

```rust
pub struct Tensor<'a> {
    // Tensor metadata and data reference
}

impl<'a> Tensor<'a> {
    pub fn shape(&self) -> &[usize];
    pub fn dtype(&self) -> Dtype;
    pub fn data(&self) -> &'a [u8];
}
```

### Dtype

Supported data types:

```rust
pub enum Dtype {
    F32,   // 32-bit float
    F16,   // 16-bit float
    BF16,  // bfloat16
    I64,   // 64-bit signed integer
    I32,   // 32-bit signed integer
    U8,    // 8-bit unsigned integer
    BOOL,  // Boolean
}
```

## Performance Considerations

### When to Use Each Loading Method

| Method | Use Case | Performance | Memory |
|--------|----------|-------------|--------|
| `load()` | General purpose | Good | Allocates full file |
| `load_parallel()` | Large files (>100MB) | Best | Allocates full file |
| `load_sync()` | CLI tools, scripts | Good | Allocates full file |
| `load_mmap()` | Random access, large files | Variable | Zero-copy |

### Parallel Loading

Parallel loading provides benefits for:
- Large files (>100MB)
- Fast storage (NVMe SSD)
- Multi-core systems

Chunk count guidelines:
- Match CPU core count for balanced load
- Typical range: 4-16 chunks
- Test with your specific workload

```rust
// Auto-detect core count
let cores = num_cpus::get();
let tensors = safetensors::load_parallel("model.safetensors", cores).await?;
```

### Memory-Mapped I/O

Benefits:
- Zero-copy loading
- OS manages memory
- Good for random access

Drawbacks:
- Page faults on first access
- Not suitable for sequential reads
- Memory pressure can cause evictions

Use mmap when:
- File is larger than RAM
- You only need specific tensors
- Random access pattern

## Error Handling

All operations return `Result<T, SafeTensorError>`:

```rust
use tensor_store::safetensors::SafeTensorError;

match safetensors::load("model.safetensors").await {
    Ok(tensors) => process(tensors),
    Err(SafeTensorError::Io(e)) => {
        eprintln!("I/O error: {}", e);
    }
    Err(SafeTensorError::InvalidFormat(msg)) => {
        eprintln!("Invalid format: {}", msg);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Examples

### Convert PyTorch to SafeTensors

```rust
use tensor_store::safetensors::{SafeTensorsWriter, TensorView, Dtype};

async fn convert_model() -> Result<()> {
    let mut writer = SafeTensorsWriter::new();

    // Load from PyTorch (pseudo-code)
    let pytorch_tensors = load_pytorch("model.pt")?;

    for (name, tensor) in pytorch_tensors {
        writer.add_tensor(
            &name,
            TensorView {
                dtype: Dtype::F32,
                shape: tensor.shape().to_vec(),
                data: tensor.as_bytes(),
            }
        )?;
    }

    writer.write_to_file("model.safetensors").await?;
    Ok(())
}
```

### Inspect Model Tensors

```rust
async fn inspect_model(path: &str) -> Result<()> {
    let tensors = safetensors::load(path).await?;

    println!("Model contains {} tensors:", tensors.names().len());
    for name in tensors.names() {
        let tensor = tensors.tensor(name).unwrap();
        println!("  {}: {:?} {:?}",
                 name,
                 tensor.shape(),
                 tensor.dtype());
    }

    Ok(())
}
```

### Extract Single Tensor

```rust
async fn extract_tensor(path: &str, tensor_name: &str) -> Result<Vec<u8>> {
    let tensors = safetensors::load_mmap(path)?;
    let tensor = tensors.tensors()
        .tensor(tensor_name)
        .ok_or("Tensor not found")?;

    Ok(tensor.data().to_vec())
}
```

## Testing

```bash
# Run safetensors tests
cargo test safetensors

# Run with specific test
cargo test safetensors::reader::tests

# Benchmark safetensors loading
cargo bench --bench safetensors
```

## References

- [SafeTensors Format Specification](https://github.com/huggingface/safetensors)
- [HuggingFace SafeTensors Library](https://github.com/huggingface/safetensors)
