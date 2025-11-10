# SafeTensors to ServerlessLLM Conversion Plan

## Overview

This plan outlines the implementation of a converter that transforms SafeTensors format checkpoints into ServerlessLLM format. The conversion enables leveraging ServerlessLLM's optimized multi-threaded loading pipeline while maintaining compatibility with existing SafeTensors-based workflows.

## Format Analysis

### SafeTensors Format

- **Structure**: JSON header + raw tensor data buffer
- **Header**: Contains tensor metadata (name, dtype, shape, offsets)
- **Data**: Single contiguous buffer with all tensor data
- **Key Types**:

  ```rust
  pub enum Dtype {
      F32, F16, BF16, F64, I32, I16, I8, I64, U32, U16, U8, U64, BOOL, F8_E4M3, F8_E5M2
  }
  ```

### ServerlessLLM Format

- **Structure**: JSON index + multiple partitioned binary files
- **Files**:
  - `tensor_index.json`: Tensor metadata and file mappings
  - `tensor.data_0`, `tensor.data_1`, ...: Binary tensor data partitions
- **Index Entry**: `(offset, size, shape, stride, dtype)`
- **Partitioning**: Configurable number of partitions for parallel I/O

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 ServerlessLLM Writer Backend

**Location**: `tensor_store/src/writers/backends/`

- **async_io.rs**: Tokio-based file writing operations
- **io_uring.rs**: io_uring-based writing for Linux (zero-copy)
- **Operations**:
  - `write_all(path, data)` - Write complete buffer
  - `write_range(path, offset, data)` - Write at specific offset
  - `create_file(path)` - Create new file
  - `truncate_file(path, size)` - Preallocate file

#### 1.2 ServerlessLLM Format Writer

**Location**: `tensor_store/src/writers/formats/serverlessllm/`

- **mod.rs**: Public API and format orchestration
- **index.rs**: JSON index file writer
- **partition.rs**: Binary partition file writer
- **Key Functions**:
  - `write_model(output_dir, tensors, partition_count)`
  - `create_tensor_index(tensors)` -> JSON structure
  - `partition_tensors(tensors, partition_count)` -> Assignment logic

### Phase 2: SafeTensors Parser

#### 2.1 SafeTensors Reader Integration

**Location**: `tensor_store/src/writers/formats/safetensors/`

- **mod.rs**: SafeTensors format reader for conversion
- **Key Functions**:
  - `parse_safetensors(path)` -> Extract tensor metadata and data
  - `extract_tensor_views(safetensors)` -> Iterator over tensor views
  - `convert_dtype(safetensors_dtype)` -> Map to ServerlessLLM dtype strings

#### 2.2 Dtype Mapping

```rust
// SafeTensors Dtype -> ServerlessLLM string
F32 => "torch.float32"
F16 => "torch.float16"
BF16 => "torch.bfloat16"
F64 => "torch.float64"
I32 => "torch.int32"
I16 => "torch.int16"
I8 => "torch.int8"
I64 => "torch.int64"
U32 => "torch.uint32"
U16 => "torch.uint16"
U8 => "torch.uint8"
U64 => "torch.uint64"
BOOL => "torch.bool"
```

### Phase 3: Conversion Orchestration

#### 3.1 High-Level Converter

**Location**: `tensor_store/src/writers/converters/safetensors_to_serverlessllm.rs`

- **Function**: `convert_safetensors_to_serverlessllm(input_path, output_dir, partition_count)`
- **Steps**:
  1. Parse SafeTensors file
  2. Extract tensor metadata and data
  3. Convert dtypes and shapes
  4. Partition tensors across files
  5. Write ServerlessLLM format

#### 3.2 Partitioning Strategy

- **Algorithm**: Round-robin assignment by tensor size (largest first)
- **Optimization**: Balance I/O load across partitions
- **Formula**: `partition_id = tensor_index % partition_count`

### Phase 4: Integration and Testing

#### 4.1 Public API

**Location**: `tensor_store/src/lib.rs`

- Add `convert_safetensors_to_serverlessllm` to public API
- Update `writers` module exports

#### 4.2 Testing

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end conversion verification
- **Performance Tests**: Compare loading speeds between formats

## Technical Details

### Memory Management

- **Streaming**: Process tensors without loading entire SafeTensors file
- **Zero-copy**: Use memory mapping where possible
- **Chunked Writing**: Write large tensors in chunks to avoid memory pressure

### Error Handling

- **Validation**: Check tensor data integrity during conversion
- **Recovery**: Handle partial conversion failures
- **Logging**: Detailed progress reporting for large models

### Performance Optimizations

- **Parallel I/O**: Utilize multiple partitions for concurrent writing
- **Async Operations**: Non-blocking file operations
- **Buffering**: Optimal buffer sizes for different storage types

## Implementation Order

1. **Phase 1**: Writer backends (async_io, io_uring)
2. **Phase 1**: ServerlessLLM format writer
3. **Phase 2**: SafeTensors parser
4. **Phase 3**: Conversion orchestration
5. **Phase 4**: Integration and testing

## Success Criteria

- **Functional**: Convert any valid SafeTensors file to ServerlessLLM format
- **Performance**: Conversion time < 2x original file size / bandwidth
- **Compatibility**: Generated files loadable by ServerlessLLM
- **Memory**: Peak memory usage < 2x largest tensor size
- **Reliability**: Comprehensive error handling and validation

## Dependencies

- **External**: `serde_json` for JSON handling
- **Internal**: Existing `loaders` infrastructure for reference
- **Platform**: Linux for io_uring optimizations

## Risk Mitigation

- **Incremental**: Each phase delivers working functionality
- **Testing**: Extensive validation at each step
- **Fallbacks**: Graceful degradation on unsupported platforms
- **Documentation**: Comprehensive inline documentation</content>

