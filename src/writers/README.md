# Writers Module

This module provides functionality to **serialize and write** checkpoint formats to disk.

## Purpose

Writers are responsible for:
- **Serializing** checkpoint data to specific formats
- **Writing** format-specific file structures
- **Creating** output files and directories
- **Format-specific logic** for writing different checkpoint types

## Architecture Principle

**Writers should ONLY write their own format.** They should not:
- Convert between formats
- Know about other checkpoint formats
- Perform any transformation logic

## Module Structure

```
writers/
├── safetensors.rs    # SafeTensors format writer
├── serverlessllm.rs  # ServerlessLLM format writer
├── tensorstore.rs    # TensorStore format writer
└── mod.rs            # Module exports
```

## Key Interfaces

### SafeTensors Writer
```rust
use tensor_store::writers::{SafeTensorsWriter, MetadataMap, TensorView, Dtype};

let writer = SafeTensorsWriter::new();
let tensor = TensorView::new(Dtype::F32, vec![1, 1], &[0u8; 4]).unwrap();
let _buffer = writer.write_to_buffer([("weight", tensor.clone())], None)?;
writer.write_to_file([("weight", tensor)], MetadataMap::default(), "model.safetensors")?;
```

### ServerlessLLM Writer
```rust
use tensor_store::writers::{ServerlessLlmWriter, TensorEntry};
use std::collections::HashMap;

let writer = ServerlessLlmWriter::new();
let tensors: HashMap<String, TensorEntry> = HashMap::new();

writer.write_index("tensor_index.json", &tensors).await?;
writer.write_partition("tensor.data_0", 0, &[0u8; 1024]).await?;
```

### TensorStore Writer
```rust
use tensor_store::writers::{TensorStoreWriter, TensorStoreIndexEntry};

let writer = TensorStoreWriter::new();
let entries = vec![TensorStoreIndexEntry::default()];

writer.write_index("model.index", &entries).await?;
writer.write_shard("shard_0.bin", 0, &[0u8; 4096]).await?;
```

### Backend I/O Operations
```rust
use tensor_store::backends;

// Async file operations
backends::async_io::write_all(path, data).await?;
backends::async_io::write_range(path, offset, data).await?;

// io_uring operations (Linux only)
backends::io_uring::write_all(path, data).await?;
```

## Usage Pattern

```rust
// 1. Converters prepare data in target format
let serverlessllm_tensors = converters::prepare_serverlessllm_data(safetensors_data);

// 2. Use appropriate writer for output format
let writer = writers::ServerlessLlmWriter::new();
writer.write_index(output_path, &serverlessllm_tensors).await?;

// 3. Write partition files as needed
for (partition_id, data) in partitions {
    let partition_path = format!("tensor.data_{}", partition_id);
    writer.write_partition(&partition_path, data).await?;
}
```

## Backend Selection

The module provides multiple I/O backends optimized for different scenarios:

- **async_io**: General-purpose async file operations using Tokio
- **io_uring**: High-performance zero-copy operations on Linux systems

Backends are automatically selected based on platform and availability.
