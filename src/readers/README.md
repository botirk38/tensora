# Readers Module

This module provides functionality to **parse and deserialize** checkpoint formats from disk.

## Purpose

Readers are responsible for:
- **Parsing** checkpoint format metadata and data
- **Deserializing** format-specific structures
- **Providing typed access** to checkpoint contents
- **Format-specific logic** for reading different checkpoint types

## Architecture Principle

**Readers should ONLY parse their own format.** They should not:
- Convert between formats
- Know about other checkpoint formats
- Perform any transformation logic

## Module Structure

```
readers/
├── safetensors.rs     # SafeTensors format parser (re-exports safetensors crate)
├── serverlessllm.rs   # ServerlessLLM index.json parser
├── tensorstore.rs     # TensorStore index parser
└── mod.rs            # Module exports
```

## Key Interfaces

### SafeTensors Reader
```rust
use tensor_store::readers::safetensors;

// Re-exported types from safetensors crate
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

// Convenience loading function
let data = safetensors::load_and_parse("model.safetensors").await?;
let tensors = SafeTensors::deserialize(&data)?;
```

### ServerlessLLM Reader
```rust
use tensor_store::readers::serverlessllm;

// Parse index.json
let index = serverlessllm::parse_index("tensor_index.json").await?;
```

## Usage Pattern

```rust
// 1. Use appropriate reader for input format
let tensors = readers::safetensors::load_and_parse(path).await?;

// 2. Access parsed data through reader interfaces
for name in tensors.names() {
    let view = tensors.tensor(name)?;
    println!("{}: {:?} ({})", name, view.shape(), view.dtype());
}

// 3. Pass parsed data to converters for transformation
converters::safetensors_to_serverlessllm::convert(tensors, output_path).await?;
```
readers/
├── formats/           # Format-specific parsers
│   ├── safetensors/   # SafeTensors format parser (re-exports safetensors crate)
│   ├── serverlessllm/ # ServerlessLLM index.json parser
│   └── tensorstore/   # TensorStore index parser
└── mod.rs            # Module exports
```

## Key Interfaces

### SafeTensors Reader
```rust
use tensor_store::readers::formats::safetensors;

// Re-exported types from safetensors crate
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

// Convenience loading function
let data = safetensors::load_and_parse("model.safetensors").await?;
let tensors = SafeTensors::deserialize(&data)?;
```

### ServerlessLLM Reader
```rust
use tensor_store::readers::formats::serverlessllm;

// Parse index.json
let index = serverlessllm::parse_index("tensor_index.json").await?;
```

## Usage Pattern

```rust
// 1. Use appropriate reader for input format
let tensors = readers::formats::safetensors::load_and_parse(path).await?;

// 2. Access parsed data through reader interfaces
for name in tensors.names() {
    let view = tensors.tensor(name)?;
    println!("{}: {:?} ({})", name, view.shape(), view.dtype());
}

// 3. Pass parsed data to converters for transformation
converters::safetensors_to_serverlessllm::convert(tensors, output_path).await?;
```