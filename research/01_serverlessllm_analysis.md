# ServerlessLLM Storage Architecture Deep Dive

## Overview

ServerlessLLM implements a sophisticated multi-tier storage system that achieves 5-10x faster loading speeds compared to traditional approaches through a combination of optimized checkpoint formats and multi-threaded I/O pipelines.

## Multi-tier Storage Engine Architecture

### Core Components

#### Memory Pool Management
- **Configurable Size**: `--mem-pool-size` parameter (default recommendations vary by use case)
- **Chunk-based Allocation**: Default 32MB chunks (`--chunk-size` parameter)
- **Pinned Memory**: CUDA pinned memory via `PinnedMemoryPool` for zero-copy GPU transfers
- **LRU Eviction**: Least Recently Used cache eviction policy

#### Storage Hierarchy
```
┌─────────────────┐
│      DRAM       │ ← Memory Pool (Pinned)
├─────────────────┤
│       SSD       │ ← Primary Storage
├─────────────────┤
│       HDD       │ ← Cold Storage
└─────────────────┘
```

### Checkpoint Format Analysis

ServerlessLLM uses a custom checkpoint format optimized for parallel loading:

```
ServerlessLLM Model Directory:
├── tensor_index.json              # Tensor metadata and file mapping
├── tensor.data_0                  # Binary tensor data (partition 0)
├── tensor.data_1                  # Binary tensor data (partition 1)
├── tensor.data_N                  # Additional partitions
├── config.json                    # Model configuration
├── no_split_modules.json          # Non-splittable modules
├── tied_no_split_modules.json     # Shared parameters
└── rank_*/                        # vLLM per-GPU partitions
    ├── tensor.data_*
    └── tensor_index.json
```

#### Key Format Features
1. **Partitioned Storage**: Model tensors split across multiple binary files
2. **Metadata Separation**: JSON index maps tensor names to file offsets/sizes
3. **GPU-Specific Partitions**: Pre-computed partitions for vLLM multi-GPU setups
4. **Raw Binary Data**: No compression, enabling direct memory mapping

### Multi-threaded Pipeline Implementation

#### Core Classes and Methods

From `sllm_store/csrc/sllm_store/model.h` and `checkpoint_store.cpp`:

```cpp
class Model {
    int ToHost(int num_threads);          // Parallel disk → pinned memory
    int ToGpu(const std::string& replica_uuid,
              const MemPtrListMap& device_ptrs,
              const MemCopyChunkList& chunks);  // Pinned memory → GPU
    int WaitInHost();                     // Synchronization
    int WaitInGpu(const std::string& replica_uuid);
};
```

#### Threading Architecture

1. **Disk I/O Phase**:
   - Configurable thread count via `--num-thread` parameter
   - Parallel reads of partitioned tensor files
   - Data loaded into pinned memory pool

2. **GPU Transfer Phase**:
   - Uses CUDA IPC handles for direct memory transfer
   - Managed by `GpuReplica` structures
   - Bypasses CPU memory copies

3. **Synchronization**:
   - `std::condition_variable` for thread coordination
   - `MemoryState` tracking (UNINITIALIZED → HOST → GPU)

### Performance Characteristics

#### Measured Improvements
- **5-10x faster** than safetensors and PyTorch checkpoint loader
- **5-100x lower** startup latency than Ray Serve and KServe
- Successfully saturates storage bandwidth on modern NVMe drives

#### Bottleneck Analysis

From the codebase analysis, identified bottlenecks include:

1. **Thread Management Overhead**:
   ```cpp
   // From checkpoint_store.cpp:84
   LOG(INFO) << "I/O threads: " << num_thread
             << ", chunk size: " << chunk_size / MB << "MB";
   ```
   - CPU cycles spent on thread creation and destruction
   - Context switching overhead with high thread counts
   - Synchronization primitive overhead (`std::mutex`, `std::condition_variable`)

2. **Sequential Model Processing**:
   ```python
   # From sllm_store/server.py - models processed one at a time
   async def LoadModelAsync(self, request, context):
       # Each model loading is sequential despite internal parallelization
   ```

3. **Memory Copy Stages**:
   - Disk → Pinned Memory (multi-threaded)
   - Pinned Memory → GPU Memory (CUDA IPC)
   - Multiple copy operations introduce latency

4. **I/O Thread Tuning**:
   - Too few threads: underutilize disk bandwidth
   - Too many threads: context switching overhead
   - Optimal count depends on storage hardware characteristics

### Integration with ML Frameworks

#### Transformers Backend
```python
# From sllm_store/transformers.py
def load_model(model_path, fully_parallel=False):
    if fully_parallel:
        return fully_parallel_load(model_path)
    # Falls back to "best-effort" parallelization
```

#### vLLM Integration
- Patch system adds `SERVERLESS_LLM` as `LoadFormat` option
- Pre-computed per-GPU partitions in `rank_*/` directories
- Direct integration with vLLM's model loading pipeline

### Configuration and Tuning

#### Key Parameters
```bash
sllm-store start \
    --storage-path /path/to/models \
    --mem-pool-size 8GB \
    --chunk-size 32MB \
    --num-thread 16 \
    --port 8073
```

#### Hardware Considerations
- **NVMe SSDs**: Higher thread counts (16-32) for maximum bandwidth
- **SATA SSDs**: Lower thread counts (4-8) to avoid overhead
- **Network Storage**: Reduced thread counts, larger chunk sizes

## Comparison with Traditional Approaches

### PyTorch Default Loading
- Sequential file reading
- Python pickle deserialization overhead
- Multiple memory copies
- No parallel I/O

### Safetensors
- Memory-mapped file access
- JSON metadata parsing
- Limited parallelization
- Alignment issues with GPU kernels

### ServerlessLLM Advantages
- Parallel disk I/O from multiple partitions
- Optimized binary format
- Direct GPU memory transfer
- Configurable memory pooling

## Identified Optimization Opportunities

1. **Replace Threading with io_uring**: Eliminate thread overhead while maintaining parallelism
2. **Batch I/O Operations**: Use vectored I/O for multiple tensor loading
3. **Improved Memory Alignment**: Address GPU kernel alignment requirements
4. **NUMA Awareness**: Optimize memory allocation for multi-GPU systems
5. **Async Pipeline**: Overlap disk I/O with GPU transfers

These findings form the foundation for TensorStore's io_uring-native approach to achieve similar performance gains with better CPU efficiency.