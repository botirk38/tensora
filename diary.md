# Project Development Diary

## 2025-11-10

- Organized project into modular modules (backends, converts, readers, writers). This allows us to reuse and encapsulate logic.
- Implemented mmap loader to use the OS's virtual memory system for efficient file access.

## 2025-10-29

- Implemented zero-copy parallel loading for both async_io and io_uring backends
- Key innovation: Pre-allocate single final buffer, split into non-overlapping slices for parallel tasks
- Created BufferSlice abstraction (async_io) and BorrowedVec (io_uring) for safe slice passing to async tasks
- Achieved ~50% speedup for parallel loads by eliminating 523MB of memory copies
- Added load_range() function to both backends for efficient byte-range loading (enables future sharded format support)
- Refactored codebase to format-based organization:
  - backends/ - Format-agnostic I/O implementations (async_io, io_uring)
  - formats/ - Format-specific wrappers (safetensors)
- Added comprehensive zero-copy tests for both Linux (io_uring) and portable (tokio) implementations
- All tests passing, documentation generated successfully

## 2025-10-30

- Created comprehensive writers module structure for checkpoint format conversion
- Designed modular architecture mirroring loaders pattern:
  - writers/backends/ - Format-agnostic I/O operations (async_io, io_uring)
  - writers/formats/ - Format-specific writers (SafeTensors, TensorStore, ServerlessLLM)
  - writers/converters/ - High-level format conversion orchestration
- Created detailed SafeTensors to ServerlessLLM conversion plan (SAFETENSORS_TO_SERVERLESSLLM_PLAN.md)
- Analyzed ServerlessLLM format: JSON index + multiple partitioned binary files
- Planned 4-phase implementation: infrastructure, parser, orchestration, integration
- Added writers module to lib.rs public API
- All placeholder files created with comprehensive documentation

## 2025-10-19

- Replaced custom PinnedPool with external zeropool crate for better buffer management
- Made tokio loader available on all platforms (previously Linux-only used io_uring)
- Added comprehensive benchmark suite comparing pooled vs non-pooled variants:
  - sync_safetensors (tokio loader, no pool)
  - sync_safetensors_with_pool (tokio loader, with pool)
  - io_uring_safetensors (io_uring, no pool)
  - io_uring_safetensors_with_pool (io_uring, with pool)
  - io_uring_safetensors_with_pool_pinned (io_uring, pinned memory pool)
  - tokio_safetensors (async tokio, no pool)
  - tokio_safetensors_with_pool (async tokio, with pool)
  - tokio_safetensors_with_pool_pinned (async tokio, pinned memory pool)
- Updated tokio dependency with io-util and rt-multi-thread features for proper async support
- Removed custom buffer pool implementation in favor of battle-tested external crate
- Key lesson: Don't reinvent the wheel when external crates already solve the problem well

## 2025-10-18

- Implemented high-performance buffer pool achieving 70% speedup over no-pool baseline
- Custom pool implementation beats io_uring without pool: 176ms → 52ms (3.36x faster)
- Key optimizations applied:
  - Thread-local storage for lock-free fast path
  - parking_lot::Mutex instead of std::sync::Mutex
  - First-fit (O(1)) allocation instead of best-fit (O(n))
  - Unsafe set_len() to avoid zero-filling 500MB buffers
  - Data-driven defaults: 1MB min buffer, 16 buffer max pool size
- Replaced nix crate with cross-platform region crate for mlock support
- Discovered through benchmarking that unsafe code is critical: safe resize() causes massive slowdown
- Validated that all "magic values" should be data-driven: initial 4KB min was 130,000x too small for 523MB tensor files
- Learned that premature optimization (best-fit search) can hurt more than help in uniform workloads

## 2025-10-09

- Setup io_uring loaders and basic tokio loaders, beat sync loading by 5%

## 2025-09-30

- Scaffolded initial Zig project structure with build.zig and build.zig.zon files
- Added ServerlessLLM repo to use as a reference to my own implementation
- Migrated from Zig to Rust due to WriterGate breaking changes in Zig I/O system
- Removed Zig build files and source directory
- Updated proposal.md to use Rust instead of Zig
- Completed comprehensive research phase covering ServerlessLLM architecture analysis
- Researched io_uring ecosystem - found Tokio has no built-in support, tokio-uring crate available
- Designed TensorStore format specification with 64-byte alignment to solve safetensors GPU issues
- Created research documentation structure in /research/ with 5 detailed analysis documents
- Updated roadmap.md to reflect completed research phase and MVP-focused next steps
- Clarified project scope: custom TensorStore format is for learning, real innovation is io_uring loading approach
- Updated MVP to focus on safetensors + tokio-uring vs safetensors + tokio::fs comparison
- Redesigned project scope for 300-hour constraint
- Refocused roadmap on validation rather than production system
- Updated implementation strategy to prioritize empirical testing over features
- Established Week 6 go/no-go decision point for performance validation
- Revised time estimates to be more realistic for competent engineer (300 hours → MVP in weeks 3-4, analysis weeks 5-7)

## 2025-11-10

- Major architectural refactoring: reorganized project into modular structure
- Migrated from loaders/ to backends/ architecture for better separation of concerns
- Implemented mmap loader using OS's virtual memory system for efficient file access
- Created comprehensive module structure:
  - backends/ - Format-agnostic I/O implementations (async_io, io_uring, mmap)
  - readers/ - Format-specific readers (SafeTensors, ServerlessLLM, TensorStore)
  - writers/ - Format-specific writers (SafeTensors, ServerlessLLM, TensorStore)
  - converters/ - High-level format conversion orchestration
- Added detailed README documentation for readers, writers, and converters modules
- Created SafeTensors to ServerlessLLM conversion plan (SAFETENSORS_TO_SERVERLESSLLM_PLAN.md)
- Laid foundation for complete checkpoint format conversion workflow
- Cleaned up SafeTensors reader API for better usability

## 2025-11-21

- Implemented complete ServerlessLLM format support (reader + writer)
- Established writers interface with consistent API across all formats
- Added comprehensive error handling infrastructure:
  - readers/error.rs - Reader-specific error types
  - writers/error.rs - Writer-specific error types
  - traits.rs - Shared traits for readers and writers
- Implemented SafeTensors writer with full functionality
- Implemented TensorStore writer structure
- Enhanced backends with write capabilities:
  - async_io backend now supports writing
  - io_uring backend now supports writing
  - mmap backend enhanced with write operations
- Completed SafeTensors to ServerlessLLM converter implementation
- Added serde and serde_json dependencies for JSON metadata handling
- Fixed clippy warnings for cleaner codebase
- Updated safetensors dependency to latest version (0.4.8)
- Key achievement: Project now has complete read-write cycle for all three formats
