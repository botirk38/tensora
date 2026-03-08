# ServerlessLLM Module

Implementation of ServerlessLLM format reading and writing for tensor_store.

## Overview

ServerlessLLM is a partitioned tensor storage format designed for efficient serverless LLM inference. This module provides:
- High-performance reading with partitioned data loading
- Writing tensors to ServerlessLLM format with configurable partitions
- Parallel loading of partition files

Based on the ServerlessLLM paper (OSDI '24).

## Format Specification

ServerlessLLM uses a directory-based format with partitioned data:

```
model_serverlessllm/
├── tensor_index.json       # Metadata and tensor index
├── tensor.data_0           # Partition 0
├── tensor.data_1           # Partition 1
├── tensor.data_2           # Partition 2
└── ...
```

### Index Format (`tensor_index.json`)

```json
{
  "metadata": {
    "version": "1.0",
    "num_partitions": 8
  },
  "tensors": {
    "model.embed.weight": {
      "dtype": "F32",
      "shape": [50257, 768],
      "partition": 0,
      "offset": 0,
      "size": 154664448
    },
    "model.layer0.weight": {
      "dtype": "F32",
      "shape": [768, 3072],
      "partition": 1,
      "offset": 0,
      "size": 9437184
    }
  }
}
```

## Module Structure

```
serverlessllm/
├── mod.rs     # Public API re-exports
├── types.rs   # Common types (TensorEntry)
├── reader.rs  # ServerlessLLM reading implementations
└── writer.rs  # ServerlessLLM writing implementations
```

## Reading ServerlessLLM

### Async Loading

```rust
use tensor_store::serverlessllm;

// Load entire model directory
let model = serverlessllm::load("model_serverlessllm").await?;

println!("Loaded {} tensors from {} partitions",
         model.len(),
         model.num_partitions());

// Access tensor data
for (name, tensor) in &model {
    println!("{}: {:?} {:?}",
             name,
             tensor.shape(),
             tensor.dtype());
    let data: &[u8] = tensor.data();
}

// Parse index only (without loading data)
let index = serverlessllm::parse_index("model_serverlessllm").await?;
println!("Index contains {} tensors", index.tensors().len());
```

### Synchronous Loading

```rust
use tensor_store::serverlessllm;

// Blocking load (no async runtime needed)
let model = serverlessllm::load_sync("model_serverlessllm")?;

// Parse index synchronously
let index = serverlessllm::parse_index_sync("model_serverlessllm")?;
```

### Memory-Mapped Loading

```rust
use tensor_store::serverlessllm;

// Memory-map partitions for lazy loading
let model = serverlessllm::load_mmap("model_serverlessllm")?;

// Data is loaded on access
for name in model.tensor_names() {
    let tensor = model.tensor(name).unwrap();
    // Page fault occurs here on first access
    let data = tensor.data();
}
```

## Writing ServerlessLLM

### Basic Writing

```rust
use tensor_store::serverlessllm::ServerlessLlmWriter;
use tensor_store::safetensors::Dtype;

let mut writer = ServerlessLlmWriter::new(8); // 8 partitions

// Add tensors (writer assigns partitions automatically)
let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
writer.add_tensor(
    "model.weight",
    Dtype::F32,
    vec![2, 2],
    bytemuck::cast_slice(&weights),
)?;

// Write to directory
writer.write_to_dir("output_serverlessllm").await?;
```

### Converting from SafeTensors

```rust
use tensor_store::{safetensors, serverlessllm};

async fn convert(input: &str, output: &str, partitions: usize) -> Result<()> {
    // Load SafeTensors
    let st = safetensors::load(input).await?;

    // Create ServerlessLLM writer
    let mut writer = serverlessllm::ServerlessLlmWriter::new(partitions);

    // Add all tensors
    for name in st.names() {
        let tensor = st.tensor(name).unwrap();
        writer.add_tensor(
            name,
            tensor.dtype(),
            tensor.shape().to_vec(),
            tensor.data(),
        )?;
    }

    // Write partitioned format
    writer.write_to_dir(output).await?;
    Ok(())
}
```

## Types

### ServerlessLLM

Owned ServerlessLLM data loaded into memory:

```rust
pub struct ServerlessLLM {
    // Owned tensor data with partition information
}

impl ServerlessLLM {
    pub fn len(&self) -> usize;
    pub fn num_partitions(&self) -> usize;
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)>;
}
```

### ServerlessLLMMmap

Memory-mapped ServerlessLLM (zero-copy):

```rust
pub struct ServerlessLLMMmap {
    // Memory-mapped partition files
}

impl ServerlessLLMMmap {
    pub fn tensor_names(&self) -> Vec<&str>;
    pub fn tensor(&self, name: &str) -> Option<TensorMmap>;
}
```

### ServerlessLLMIndex

Parsed index without tensor data:

```rust
pub struct ServerlessLLMIndex {
    // Index metadata
}

impl ServerlessLLMIndex {
    pub fn tensors(&self) -> &HashMap<String, TensorEntry>;
    pub fn num_partitions(&self) -> usize;
}
```

### TensorEntry

Tensor metadata from index:

```rust
pub struct TensorEntry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub partition: usize,   // Which partition file
    pub offset: usize,      // Offset within partition
    pub size: usize,        // Size in bytes
}
```

## Performance Considerations

### Partition Count

Choosing the right number of partitions:

| Partitions | Use Case | Pros | Cons |
|------------|----------|------|------|
| 1 | Small models (<1GB) | Simple | No parallelism |
| 4-8 | Medium models (1-10GB) | Good balance | Multiple files |
| 16-32 | Large models (>10GB) | Max parallelism | Many file handles |

Guidelines:
- Match partition count to available CPU cores
- More partitions = more parallel I/O
- Diminishing returns beyond 16 partitions

```rust
let cores = num_cpus::get();
let writer = ServerlessLlmWriter::new(cores.min(16));
```

### When to Use ServerlessLLM

**Advantages over SafeTensors:**
- Parallel loading of different partitions
- Can load specific tensors without reading entire file
- Better for serverless cold-start scenarios

**Disadvantages:**
- More complex format (directory vs single file)
- Multiple file handles
- More disk seeks

Use ServerlessLLM when:
- Loading only specific layers/tensors
- Parallel loading is critical
- Serverless/cold-start optimization matters

Use SafeTensors when:
- Loading entire model
- Simple single-file format preferred
- Sequential access pattern

### Memory-Mapped vs Owned

| Method | Memory Usage | Load Time | Access Pattern |
|--------|--------------|-----------|----------------|
| `load()` | Full model size | Upfront | Sequential |
| `load_mmap()` | Minimal | Lazy | Random access |

## Error Handling

```rust
use tensor_store::serverlessllm;

match serverlessllm::load("model_serverlessllm").await {
    Ok(model) => process(model),
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
        eprintln!("Directory not found");
    }
    Err(e) => {
        eprintln!("Error loading ServerlessLLM: {}", e);
    }
}
```

## Examples

### Inspect Model Structure

```rust
async fn inspect(dir: &str) -> Result<()> {
    let index = serverlessllm::parse_index(dir).await?;

    println!("ServerlessLLM Model:");
    println!("  Partitions: {}", index.num_partitions());
    println!("  Tensors: {}", index.tensors().len());

    for (name, entry) in index.tensors() {
        println!("    {}: {:?} @ partition {} offset {}",
                 name,
                 entry.shape,
                 entry.partition,
                 entry.offset);
    }

    Ok(())
}
```

### Load Specific Tensors

```rust
async fn load_layers(dir: &str, layer_prefix: &str) -> Result<Vec<Vec<u8>>> {
    let model = serverlessllm::load_mmap(dir)?;

    let mut layers = Vec::new();
    for name in model.tensor_names() {
        if name.starts_with(layer_prefix) {
            let tensor = model.tensor(name).unwrap();
            layers.push(tensor.data().to_vec());
        }
    }

    Ok(layers)
}
```

### Benchmark Partition Loading

```rust
async fn benchmark_partitions(dir: &str) -> Result<()> {
    use std::time::Instant;

    let start = Instant::now();
    let model = serverlessllm::load(dir).await?;
    let duration = start.elapsed();

    println!("Loaded {} tensors from {} partitions in {:?}",
             model.len(),
             model.num_partitions(),
             duration);

    let bytes: usize = model.iter()
        .map(|(_, t)| t.data().len())
        .sum();

    let throughput = bytes as f64 / duration.as_secs_f64() / 1_000_000_000.0;
    println!("Throughput: {:.2} GB/s", throughput);

    Ok(())
}
```

## Converting Between Formats

Use the `convert` binary:

```bash
# Convert SafeTensors to ServerlessLLM
cargo run --release --bin convert -- \
    input.safetensors \
    output_serverlessllm \
    8  # number of partitions
```

Or programmatically:

```rust
use tensor_store::{safetensors, serverlessllm};

async fn convert_format(
    input_st: &str,
    output_sllm: &str,
    partitions: usize,
) -> Result<()> {
    let st = safetensors::load(input_st).await?;
    let mut writer = serverlessllm::ServerlessLlmWriter::new(partitions);

    for name in st.names() {
        let tensor = st.tensor(name).unwrap();
        writer.add_tensor(
            name,
            tensor.dtype(),
            tensor.shape().to_vec(),
            tensor.data(),
        )?;
    }

    writer.write_to_dir(output_sllm).await?;
    Ok(())
}
```

## Testing

```bash
# Run serverlessllm tests
cargo test serverlessllm

# Benchmark serverlessllm loading
cargo bench --bench serverlessllm

# Profile loading
cargo run --bin serverlessllm_reader --release -- model_serverlessllm/
```

## References

- [ServerlessLLM Paper (OSDI '24)](https://www.usenix.org/conference/osdi24/presentation/fu)
- [ServerlessLLM GitHub](https://github.com/ServerlessLLM/ServerlessLLM)
