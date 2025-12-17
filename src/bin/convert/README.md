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

## Performance Considerations

### Partition Count Guidelines

| Model Size | Recommended Partitions | Reasoning |
|------------|----------------------|-----------|
| < 1 GB | 4 | Minimal overhead, good parallelism |
| 1-10 GB | 8 | Balanced file count and parallelism |
| 10-50 GB | 16 | Maximum parallelism for large models |
| > 50 GB | 32 | Consider storage bandwidth limits |

**Rule of thumb**: Match your CPU core count, capped at 16-32 partitions.

### Conversion Speed

Typical conversion speeds (on NVMe SSD):
- Small models (<1GB): ~2-5 seconds
- Medium models (1-10GB): ~10-30 seconds
- Large models (>10GB): ~1-2 minutes per 10GB

## Troubleshooting

### "Failed to create output directory"
Ensure the parent directory exists and you have write permissions:
```bash
mkdir -p output
cargo run --bin convert -- model.safetensors ./output 8
```

### "Out of memory"
For very large models, ensure sufficient RAM. The converter loads tensors in chunks but needs memory for partition buffers.

### "Partition files are uneven"
This is normal - tensors are distributed round-robin to partitions. Some variation in partition sizes is expected.

## Output Structure

After conversion, `output_dir` contains:
```
output_dir/
├── tensor_index.json          # Metadata (dtype, shape, partition info)
├── tensor.data_0              # Partition 0 (first 1/N of tensors)
├── tensor.data_1              # Partition 1
├── ...
└── tensor.data_{N-1}          # Last partition
```

## Platform Support

- **Linux**: Uses io_uring for high-performance async I/O
- **Other platforms**: Uses tokio runtime for async operations

## See Also

- [SafeTensors Module](../../safetensors/README.md)
- [ServerlessLLM Module](../../serverlessllm/README.md)
- [Converters Architecture](../../converters/README.md)
