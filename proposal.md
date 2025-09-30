# TensorStore: An io_uring-Native Evolution of the ServerlessLLM Storage Engine

## Aims

The aim of the project is to develop a high-performance storage engine for Large Language Model (LLM) loading using Linux's modern io_uring interface. Besides providing superior I/O throughput, the system should achieve better CPU efficiency and demonstrate measurable performance improvements over existing multi-threaded approaches through a co-designed checkpoint format and asynchronous I/O architecture.

## Background

The ServerlessLLM paper (OSDI '24) made a landmark contribution to solving LLM cold starts by introducing a fast, multi-tier storage engine with a loading-optimized checkpoint format and high-throughput I/O pipeline. While ServerlessLLM successfully saturates storage bandwidth using traditional multi-threaded, pipelined architecture, this raises the next logical research question: what is the most efficient software architecture for driving I/O at the absolute limit of modern hardware?

Multi-threaded pipelines, while effective, introduce potential bottlenecks including CPU overhead from thread creation, context switching, and synchronization primitives, as well as complexity in managing multi-stage pipelines. Modern Linux systems provide io_uring, a high-performance asynchronous I/O interface that enables kernel-offloaded I/O operations with minimal CPU overhead. TensorStore proposes that replacing the threaded pipeline with an io_uring-native event loop represents a more efficient and scalable architecture for maximizing I/O throughput.

## Prerequisites

- Strong understanding of systems programming and I/O architectures
- Proficiency in Rust programming language
- Knowledge of Linux kernel interfaces, particularly io_uring
- Understanding of machine learning model formats and loading patterns
- Experience with performance profiling and benchmarking

## Early Deliverables

### Proof of Concept Programs

- A basic io_uring integration using tokio-uring and liburing bindings demonstrating asynchronous file I/O operations
- A prototype .tensorstore format converter that transforms .safetensors files into the new layout

### Reports

- A comprehensive survey of existing model loading techniques and storage engines, with focus on ServerlessLLM
- A detailed specification of the proposed .tensorstore format and its design rationale
- Description of the prototype implementations and initial performance measurements

## Final Deliverables

### Implementation Requirements

- The system must be implemented according to modern software engineering principles with comprehensive error handling and logging
- The TensorStore format specification must be complete with a working conversion tool from popular formats (.safetensors, .bin)
- An io_uring-native loader implementation that correctly loads models into pre-allocated memory buffers
- The system should demonstrate measurable performance improvements over baseline implementations

### Reports

- A comprehensive overview of high-performance storage engines for AI workloads
- Detailed description of the TensorStore format including metadata structure, alignment requirements, and I/O optimization strategies
- Description of the io_uring implementation including event loop design, batching strategies, and memory management
- Rigorous performance evaluation comparing against PyTorch, SafeTensors, and re-implemented ServerlessLLM baselines
- Analysis of implementation challenges, design trade-offs, and software engineering process

## Extensions

- **Multi-model loading**: Support for loading multiple models concurrently using separate io_uring instances
- **Compression integration**: On-the-fly decompression during I/O to reduce storage requirements while maintaining performance
- **Network storage support**: Extension to work with network-attached storage and distributed file systems
- **Cross-platform compatibility**: Adaptation to other high-performance I/O interfaces (io_uring_prep_read_multishot, Windows IOCP)
- **Integration with ML frameworks**: Direct integration with PyTorch and other frameworks to eliminate intermediate copies
