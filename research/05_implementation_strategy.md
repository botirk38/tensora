# MVP Implementation Strategy

## Overview

This document outlines the **Minimum Viable Product (MVP)** implementation strategy for TensorStore, focusing on core functionality needed to validate the io_uring approach and demonstrate basic performance improvements.

## MVP Goals

### Primary Objective
**Validate the core hypothesis**: Can io_uring provide measurable performance improvements over safetensors for tensor loading?

### MVP Success Criteria
1. **Basic functionality**: Load tensors from a simple TensorStore format using io_uring
2. **Performance validation**: Demonstrate measurable improvement over safetensors baseline
3. **Proof of concept**: Show that the approach is technically viable

### Explicitly OUT OF SCOPE for MVP
- NUMA awareness
- Multi-GPU support
- Complex prefetching
- Production-grade error handling
- Comprehensive testing framework
- Advanced memory management
- Conversion from multiple formats

## MVP Technology Stack

### Minimal Dependencies
```toml
[package]
name = "tensorstore-mvp"
version = "0.1.0"
edition = "2021"

[dependencies]
# Core async I/O
tokio-uring = "0.5"
tokio = { version = "1.0", features = ["rt", "macros"] }

# Basic serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Memory alignment
bytemuck = { version = "1.0", features = ["derive"] }

# For testing/comparison
safetensors = "0.4"

# Error handling
thiserror = "1.0"

# Testing
criterion = { version = "0.5", features = ["html_reports"] }
```

## MVP Project Structure

```
tensorstore-mvp/
├── Cargo.toml
├── src/
│   ├── main.rs                  # CLI for testing
│   ├── lib.rs                   # Public API
│   ├── format.rs                # Simple TensorStore format
│   ├── loader.rs                # Basic async tensor loader
│   ├── converter.rs             # Safetensors → TensorStore
│   └── error.rs                 # Basic error types
├── benches/
│   └── comparison.rs            # Performance comparison
└── tests/
    └── basic.rs                 # Basic functionality tests
```

## MVP Core Components

### 1. Simple TensorStore Format

```rust
// format.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorStoreHeader {
    pub magic: [u8; 8],           // "TNSRSTR\0"
    pub version: u32,             // 1
    pub metadata_size: u64,       // Size of JSON metadata
    pub data_offset: u64,         // Where tensor data starts
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,            // "float32", "float16", etc.
    pub shape: Vec<usize>,
    pub data_offset: u64,         // Offset from data_offset in header
    pub size_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorStoreMetadata {
    pub tensors: Vec<TensorInfo>,
}

// Ensure 64-byte alignment for all tensor data
pub const TENSOR_ALIGNMENT: usize = 64;
```

### 2. Basic io_uring Loader

```rust
// loader.rs
use tokio_uring::fs::File;
use std::collections::HashMap;

pub struct TensorStoreLoader {
    file: File,
    metadata: TensorStoreMetadata,
    data_offset: u64,
}

impl TensorStoreLoader {
    pub async fn new(path: &str) -> Result<Self, crate::error::Error> {
        let file = File::open(path).await?;

        // Read header
        let header_bytes = vec![0u8; std::mem::size_of::<TensorStoreHeader>()];
        let (result, header_bytes) = file.read_at(header_bytes, 0).await;
        result?;

        let header: TensorStoreHeader = bytemuck::from_bytes(&header_bytes);

        // Read metadata
        let metadata_bytes = vec![0u8; header.metadata_size as usize];
        let (result, metadata_bytes) = file.read_at(
            metadata_bytes,
            std::mem::size_of::<TensorStoreHeader>() as u64
        ).await;
        result?;

        let metadata: TensorStoreMetadata = serde_json::from_slice(&metadata_bytes)?;

        Ok(Self {
            file,
            metadata,
            data_offset: header.data_offset,
        })
    }

    pub async fn load_tensor(&self, name: &str) -> Result<Vec<u8>, crate::error::Error> {
        let tensor_info = self.metadata.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| crate::error::Error::TensorNotFound(name.to_string()))?;

        // Allocate aligned buffer
        let mut buffer = vec![0u8; tensor_info.size_bytes];

        // Read tensor data using io_uring
        let offset = self.data_offset + tensor_info.data_offset;
        let (result, buffer) = self.file.read_at(buffer, offset).await;
        result?;

        Ok(buffer)
    }

    pub fn list_tensors(&self) -> Vec<&str> {
        self.metadata.tensors.iter().map(|t| t.name.as_str()).collect()
    }
}
```

### 3. Simple Converter

```rust
// converter.rs
use safetensors::SafeTensors;
use std::path::Path;
use tokio::fs;

pub struct SafetensorsConverter;

impl SafetensorsConverter {
    pub async fn convert(
        input_path: &Path,
        output_path: &Path,
    ) -> Result<(), crate::error::Error> {
        // Read safetensors file
        let data = fs::read(input_path).await?;
        let safetensors = SafeTensors::deserialize(&data)?;

        let mut tensors = Vec::new();
        let mut tensor_data = Vec::new();
        let mut current_offset = 0u64;

        // Process each tensor
        for (name, tensor) in safetensors.tensors() {
            let shape = tensor.shape().to_vec();
            let dtype = format!("{:?}", tensor.dtype());
            let data = tensor.data();

            // Add padding for alignment
            let aligned_offset = align_to(current_offset, TENSOR_ALIGNMENT as u64);
            let padding = aligned_offset - current_offset;

            tensor_data.extend(vec![0u8; padding as usize]);
            tensor_data.extend_from_slice(data);

            tensors.push(TensorInfo {
                name: name.to_string(),
                dtype,
                shape,
                data_offset: aligned_offset,
                size_bytes: data.len(),
            });

            current_offset = aligned_offset + data.len() as u64;
        }

        // Create metadata
        let metadata = TensorStoreMetadata { tensors };
        let metadata_json = serde_json::to_vec(&metadata)?;

        // Calculate offsets
        let header_size = std::mem::size_of::<TensorStoreHeader>();
        let data_offset = align_to(
            (header_size + metadata_json.len()) as u64,
            TENSOR_ALIGNMENT as u64
        );

        // Create header
        let header = TensorStoreHeader {
            magic: *b"TNSRSTR\0",
            version: 1,
            metadata_size: metadata_json.len() as u64,
            data_offset,
        };

        // Write file
        let mut output = Vec::new();
        output.extend_from_slice(bytemuck::bytes_of(&header));
        output.extend_from_slice(&metadata_json);

        // Pad to data section
        let padding = data_offset - output.len() as u64;
        output.extend(vec![0u8; padding as usize]);

        output.extend_from_slice(&tensor_data);

        fs::write(output_path, output).await?;
        Ok(())
    }
}

fn align_to(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}
```

### 4. Basic Error Types

```rust
// error.rs
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Invalid format")]
    InvalidFormat,
}
```

## MVP Testing Strategy

### Basic Functionality Test

```rust
// tests/basic.rs
#[tokio::test]
async fn test_convert_and_load() {
    // Create a simple safetensors file
    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors_path = temp_dir.path().join("test.safetensors");
    let tensorstore_path = temp_dir.path().join("test.tensorstore");

    // Create dummy tensor data
    create_test_safetensors(&safetensors_path).await;

    // Convert to TensorStore format
    SafetensorsConverter::convert(&safetensors_path, &tensorstore_path)
        .await
        .unwrap();

    // Load tensor using io_uring
    let loader = TensorStoreLoader::new(tensorstore_path.to_str().unwrap())
        .await
        .unwrap();

    let tensors = loader.list_tensors();
    assert!(!tensors.is_empty());

    let tensor_data = loader.load_tensor(tensors[0]).await.unwrap();
    assert!(!tensor_data.is_empty());
}
```

### Performance Comparison Benchmark

```rust
// benches/comparison.rs
use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

fn benchmark_loading(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("safetensors_loading", |b| {
        b.to_async(&rt).iter(|| async {
            // Load using safetensors
            load_with_safetensors("test.safetensors").await
        })
    });

    c.bench_function("tensorstore_loading", |b| {
        b.to_async(&rt).iter(|| async {
            // Load using TensorStore
            load_with_tensorstore("test.tensorstore").await
        })
    });
}

criterion_group!(benches, benchmark_loading);
criterion_main!(benches);
```

## MVP Development Plan

### Week 1: Core Implementation
1. **Day 1-2**: Set up project structure and dependencies
2. **Day 3-4**: Implement basic TensorStore format
3. **Day 5-7**: Implement io_uring loader

### Week 2: Testing and Validation
1. **Day 1-2**: Implement safetensors converter
2. **Day 3-4**: Create basic tests and benchmarks
3. **Day 5-7**: Performance testing and validation

### Success Metrics for MVP
- **Functionality**: Successfully convert and load tensors
- **Performance**: Show measurable improvement over safetensors (target: >20% faster)
- **Reliability**: Basic tests pass consistently

## Decision Points

### Go/No-Go Criteria
1. **Technical feasibility**: MVP works on target hardware
2. **Performance improvement**: Shows meaningful gains over baseline
3. **Development velocity**: Can implement core features within timeframe

### If MVP Succeeds
- Proceed with full TensorStore implementation
- Add advanced features (NUMA, multi-GPU, etc.)
- Implement production-grade error handling

### If MVP Fails
- Analyze bottlenecks and performance characteristics
- Consider alternative approaches (different I/O strategies)
- Document findings for future reference

This MVP strategy focuses on validating the core concept with minimal complexity, allowing for quick iteration and clear go/no-go decisions.

## Post-MVP: Next Steps if Successful

### Phase 3: Full Implementation (Weeks 4-12)

If the MVP demonstrates >20% performance improvement and validates the core approach, proceed with full implementation:

#### 3.1 Production TensorStore Format
- Enhanced format with compression integration
- Chunk-based storage for large tensors
- Format versioning and backward compatibility
- Advanced validation and error recovery

#### 3.2 Advanced io_uring Engine
- Vectored I/O operations for batch loading
- Intelligent prefetching based on access patterns
- NUMA-aware memory allocation
- Comprehensive error handling and recovery mechanisms

#### 3.3 Multi-GPU and Scaling Support
- Concurrent model loading across multiple GPUs
- Priority-based loading scheduling
- Resource isolation between models
- Memory sharing for common layers

### Phase 4: Production Readiness (Weeks 13-16)

#### 4.1 Framework Integration
- PyTorch tensor integration
- HuggingFace Transformers adapter
- Python bindings using PyO3
- C FFI for broader compatibility

#### 4.2 Production Testing
- Comprehensive performance benchmarking
- Comparison against ServerlessLLM baseline
- Production workload validation
- Memory and CPU efficiency analysis

### Phase 5: Advanced Features (Weeks 17-24)

#### 5.1 Cross-Platform Support
- Windows IOCP backend implementation
- macOS kqueue support
- Runtime I/O backend selection

#### 5.2 Enterprise Features
- Network storage support (NFS, S3)
- Distributed loading across nodes
- Monitoring and observability integration
- Security and access control

## Post-MVP: Alternative Paths if Unsuccessful

### If Performance Gains < 20%
1. **Analyze bottlenecks**: Profile to identify where gains are lost
2. **Optimize implementation**: Focus on critical performance paths
3. **Adjust targets**: Consider lower but still meaningful improvements
4. **Hybrid approach**: Combine io_uring with threading for optimal performance

### If Technical Issues Block Progress
1. **Fallback strategy**: Implement tokio::fs fallback for compatibility
2. **Alternative runtimes**: Evaluate monoio or glommio as alternatives
3. **Scope reduction**: Focus on specific use cases where io_uring excels
4. **Documentation**: Document findings for future io_uring adoption

### If Ecosystem Issues Arise
1. **Fork tokio-uring**: Take ownership of maintenance if needed
2. **Contribute upstream**: Fix issues and contribute back to ecosystem
3. **Build coalition**: Work with other users to share maintenance burden
4. **Timing consideration**: Wait for ecosystem maturity if needed

This staged approach ensures that resources are invested incrementally based on validated success at each phase.