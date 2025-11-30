# convert

Converts SafeTensors model files to ServerlessLLM format with partitioned storage.

## Overview

This binary converts a SafeTensors model file into the ServerlessLLM format, which partitions tensor data across multiple files for efficient loading and parallel access.

## Usage

```bash
cargo run --bin convert -- <input.safetensors> <output_dir> [partition_count]
```

### Arguments

- `<input.safetensors>` - Path to the input SafeTensors file
- `<output_dir>` - Directory where converted files will be written
- `[partition_count]` - (Optional) Number of partitions to create. Defaults to the number of CPU cores

### Output

The conversion produces:
- `tensor_index.json` - Index file mapping tensor names to partition locations
- `tensor.data_0` through `tensor.data_{N-1}` - Partitioned tensor data files

## Examples

Convert with automatic partition count:
```bash
cargo run --bin convert -- model.safetensors ./output
```

Convert with specific partition count:
```bash
cargo run --bin convert -- model.safetensors ./output 8
```

## Platform Support

- **Linux**: Uses io_uring for high-performance async I/O
- **Other platforms**: Uses tokio runtime for async operations
