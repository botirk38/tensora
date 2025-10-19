# Project Development Diary

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

## 2025-10-09

- Setup io_uring loaders and basic tokio loaders, beat sync loading by 5%

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
