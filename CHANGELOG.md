# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with 129+ tests
- GitLab CI/CD pipeline with build, test, lint, and security checks
- Demo binary (885 lines) for format comparison demonstrations
- Python script for downloading HuggingFace models and converting to ServerlessLLM
- Profiler binary for debugging performance regressions
- TESTING_PROGRESS.md for tracking test coverage

### Changed
- Refactored ServerlessLLM code to eliminate duplication
- Flattened repository structure for simplified project layout
- Updated dependencies with cargo-outdated
- Enhanced CI configuration with dasel installation and Rust image updates

### Fixed
- Race condition in load_parallel function
- Flaky CI tests caused by shared state between tests
- Various clippy warnings caught by automated linting

## [0.1.0] - 2025-09-30

Initial release of tensor_store library.

### Added

#### Core Features
- High-performance tensor I/O library optimized for Linux io_uring
- Cross-platform support with Tokio fallback for non-Linux systems
- Jemalloc allocator for improved memory management

#### Format Support
- **SafeTensors** format reader and writer
- **ServerlessLLM** format reader and writer with partitioned data
- Format conversion utilities (SafeTensors ↔ ServerlessLLM)

#### I/O Backends
- **io_uring backend** (Linux 5.1+) for zero-copy I/O with parallel loading
- **Tokio async backend** for cross-platform async I/O
- **Synchronous backend** using std::fs for baseline comparison
- **Memory-mapped backend** (mmap) for efficient file access via OS virtual memory
- **O_DIRECT backend** for direct I/O bypassing kernel page cache

#### Performance Optimizations
- Zero-copy parallel loading with pre-allocated buffers
- Buffer pool with thread-local storage for lock-free fast path
- Fixed buffer pools for kernel-registered buffers
- Batched io_uring operations for improved throughput
- File handle reuse across parallel reads
- 512-byte alignment support for O_DIRECT

#### Binary Applications
- `safetensors_reader` - Profile SafeTensors file loading
- `serverlessllm_reader` - Profile ServerlessLLM format loading
- `profile` - General profiling harness
- `demo` - Demonstration application showcasing library features
- `convert` - SafeTensors to ServerlessLLM converter

#### Testing & Benchmarks
- Criterion-based benchmark suite comparing all backends
- Property-based testing with proptest
- Comprehensive unit and integration tests
- Test fixtures with real model data

#### Documentation
- Complete API documentation with examples
- README with usage examples and platform support matrix
- Module-level documentation for all major components
- Development diary documenting design decisions

#### Project Infrastructure
- GitLab CI/CD with multi-platform testing
- Clippy linting configuration
- Security auditing with cargo-audit
- Cross-platform build support (Linux, macOS, Windows)

### Technical Highlights

#### Zero-Copy Architecture
- Eliminated 523MB of memory copies through BufferSlice abstraction
- 50% speedup by pre-allocating final buffer and splitting into non-overlapping slices
- Safe abstraction over unsafe pointer manipulation

#### Buffer Pool
- 70% speedup over no-pool baseline
- 3.36x improvement: 176ms → 52ms for pooled operations
- O(1) first-fit allocation strategy
- Data-driven defaults: 1MB minimum buffer, 16 buffer max pool size
- parking_lot::Mutex for reduced lock contention

#### O_DIRECT Support
- Bypasses kernel page cache for large file loading
- Handles 512-byte alignment constraints automatically
- Batch processing module for efficient grouped operations
- OwnedAlignedBuffer abstraction for safe aligned memory

#### Cross-Platform Design
- Conditional compilation for Linux-specific features
- Graceful fallback to Tokio on non-io_uring platforms
- Platform-specific optimizations while maintaining portable API

### Performance Benchmarks

Preliminary benchmarks on Linux 5.1+ (523MB SafeTensors file):
- io_uring parallel with zero-copy: **~5% faster** than tokio::fs
- Buffer pool with io_uring: **70% faster** than non-pooled
- Zero-copy approach: **50% faster** than naive copy-based parallel loading
- O_DIRECT with batching: competitive with page cache for large files

### Platform Support

| Platform | Backend | Features |
|----------|---------|----------|
| Linux 5.1+ | io_uring | Zero-copy I/O, parallel loading, fixed buffers, O_DIRECT |
| Linux (older) | Tokio | Standard async I/O |
| macOS | Tokio | Standard async I/O |
| Windows | Tokio | Standard async I/O |

### Requirements

- Rust 2024 edition
- Linux kernel 5.1+ for io_uring features
- Tokio runtime for non-Linux platforms

### Academic Context

This library is part of a Final Year Project (CS3821) at Royal Holloway, University of London, investigating whether io_uring can provide measurable performance improvements for LLM model loading compared to traditional multi-threaded approaches.

**Research Goal**: Determine if modern async I/O interfaces can exceed multi-threaded performance for tensor loading workloads.

### License

MIT License

---

## Version History Summary

- **[0.1.0]** - 2025-09-30: Initial release with complete format support, multiple I/O backends, comprehensive benchmarks
- **[Unreleased]** - Ongoing: Enhanced testing, CI/CD, tooling, and developer experience improvements
