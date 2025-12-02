# Project Development Diary

## Project Context and Evolution

This diary documents the development of **TensorStore**, a research project exploring whether Linux's modern io_uring interface can provide >20% performance improvement for Large Language Model tensor loading compared to traditional multi-threaded approaches. The project was designed as a Final Year Project (CS3821) at Royal Holloway, University of London, with a strict **300-hour constraint** across approximately 15 weeks.

### The Original Plan

The project was structured as a hypothesis-driven research effort with four distinct phases:

**Phase 1: Research** (Weeks 1-2, 40 hours) - Analyze ServerlessLLM architecture, research io_uring ecosystem, design TensorStore format specification. *This phase was completed successfully.*

**Phase 2: Core Validation** (Week 3, 20 hours) - Build minimal implementations to prove or disprove the core hypothesis: Can io_uring exceed multi-threaded performance by >20%? This was designed as a go/no-go checkpoint with a clear deliverable: benchmark showing performance difference.

**Phase 3: Analysis & Documentation** (Week 4, 30 hours) - Deep analysis of results, technical report with go/no-go recommendation for advanced features.

**Phase 4: Advanced Features** (Weeks 5-15, 210 hours) - **CONDITIONAL**: Only proceed if Phase 2 shows >20% improvement. Build production-ready features like vectored I/O, framework integration, NUMA optimization.

### The Core Hypothesis

Can io_uring's kernel-offloaded operations eliminate CPU overhead from thread context switching and synchronization, achieving measurably better performance than traditional multi-threaded approaches? The ServerlessLLM paper (OSDI '24) demonstrated that multi-threaded pipelines can saturate storage bandwidth, but this raises the question: **what is the most efficient software architecture for driving I/O at the absolute limit of modern hardware?**

### What Actually Happened

As this diary will show, the project evolved significantly beyond the minimal validation plan. Rather than stopping at the Week 3 checkpoint with a simple benchmark comparison, development continued through extensive implementation work including full format readers, writers, converters, comprehensive testing infrastructure, O_DIRECT support, and production-quality engineering practices.

This diary reflects on both the technical innovations discovered during development and the process insights about how research projects evolve when theoretical plans meet practical implementation. It documents not just achievements, but the reasoning behind decisions, the lessons learned from mistakes, and the tension between hypothesis validation and implementation learning.

The entries below are organized chronologically, showing the progression from initial research through iterative technical refinements to production-quality system design. Each entry includes reflection on what was learned, why decisions were made, and how they connect to the original project goals.

## 2025-12-02

Refactored ServerlessLLM code to eliminate duplication and remove unused functions. Added io backend tests bringing test count higher. Various CI fixes including dasel installation, cargo outdated version updates, and rust image updates. Flattened repo structure.

The refactoring was cleanup work - removed functions that weren't being used and consolidated duplicate code. The repo flattening (moving from nested structure to flat) was about simplifying the project layout.

CI fixes were maintenance - keeping dependencies updated and fixing broken pipeline steps. Not exciting work but necessary to keep the build green.

## 2025-11-29

Added 104+ tests with TESTING_PROGRESS.md to track coverage, set up full GitLab CI/CD pipeline (build, test, lint, security), wrote demo binary (885 lines) for format comparison, and added 200+ lines of documentation.

This wasn't in the original plan at all. After fixing that race condition in load_parallel, I realized I had no way to be sure it actually worked or that I wouldn't break it later. The codebase had grown way beyond a simple benchmark - there was O_DIRECT support, parallel loading, buffer pools, all with subtle correctness requirements.

The CI investment made sense after I manually forgot to run clippy before a commit and introduced warnings. Automating it meant the computer would catch what I'd forget. Same with cargo-audit - caught a vulnerable dependency I didn't know about.

The demo binary came from needing to show my supervisor actual performance numbers. I kept re-running benchmarks manually and realized I needed something reproducible with proper warmup runs and statistics. Cold cache effects matter when you're claiming performance improvements.

Big lesson: testing felt overwhelming until I broke it down in TESTING_PROGRESS.md. Phase 1: 46 tests. Phase 2: 72 tests. Suddenly it was manageable. Just writing it down made it feel doable.

The CI pipeline had flaky tests initially because I was using shared state between tests. Fixed it with tempfile for proper isolation. Learned that test code needs the same quality standards as production code.

This reflects how the project evolved beyond minimal validation. If I'd stopped at a simple benchmark in Week 3, I wouldn't need 129 tests. But once you build a real system, you need confidence it works correctly.

## 2025-11-25

Implemented O_DIRECT support - added new odirect.rs backend for direct I/O, refactored async_io, io_uring, and sync_io to support it, added batch processing module, improved ServerlessLLM parallel loading.

O_DIRECT bypasses the kernel page cache entirely. For large model loading (multi-GB files), the page cache is just overhead - you're reading data once into memory, not repeatedly accessing it. Why cache something you'll never read again?

The implementation required handling alignment constraints (O_DIRECT requires 512-byte aligned buffers and offsets). This drove the odirect.rs abstractions for align_to_block(), alloc_aligned(), and OwnedAlignedBuffer. Getting alignment wrong results in EINVAL errors from the kernel.

Added batch processing because O_DIRECT works best with larger I/O operations. Small reads perform poorly. The batch.rs module groups multiple read requests together and handles them efficiently.

This is pretty far from the original Phase 2 "minimal validation" plan. At this point I was optimizing for real-world performance, not just testing the hypothesis. O_DIRECT is something you'd use in production, not in a research prototype.

The refactoring was substantial because O_DIRECT support needed to thread through the entire backend architecture. In retrospect, designing for this from the start would have been cleaner, but you don't know what optimizations matter until you measure.

## 2025-11-24

Added buffer pool builder with presets (small/medium/large workloads), implemented profiler binary, created script to download HuggingFace models and convert to ServerlessLLM, applied linter fixes across scripts.

The builder pattern for BufferPool came from frustration with magic numbers. Earlier I'd learned that my initial 4KB minimum buffer was 130,000x too small. Now I wanted an API that made good defaults easy and bad defaults hard. Presets like "large workload" encode the learnings from benchmarking.

The profiler binary was for debugging performance regressions. Standard benchmarks tell you what's fast/slow, but profiler integration lets you see where CPU time goes. This became essential when investigating why certain operations were slower than expected.

The HuggingFace download script was practical necessity - I needed real model files to test with, not just synthetic data. Real .safetensors files have different characteristics than random bytes. The script automated what I was doing manually.

## 2025-11-23

Decoupled mmap and sync I/O backends into separate modules.

This was pure refactoring - no new functionality. The mmap backend was tangled with sync I/O code, making it hard to modify one without affecting the other. Separating them let me optimize each independently.

Clean separation of concerns. Should have done this earlier.

## 2025-11-22

Parallelized the SafeTensors to ServerlessLLM converter using `load_parallel()` and `futures::future::try_join_all`. Created shared types/ module extracting TensorEntry into one place (DRY principle).

The converter was painfully slow. Sequential loading followed by sequential writing meant the conversion time scaled linearly with model size. Parallelizing both reads and writes made it actually usable for large models.

The types/ refactoring was motivated by a bug: reader and writer had different TensorEntry definitions that were supposed to be the same. Found out when serialization broke. Extracting to a shared module fixed the immediate bug and prevented future drift.

DRY isn't just about reducing code duplication - it's about having one source of truth. If TensorEntry changes, it changes everywhere consistently.

## 2025-11-21

Implemented complete ServerlessLLM format support (reader + writer), established writers interface consistent across all formats, added error handling infrastructure, implemented SafeTensors writer, implemented TensorStore writer structure, enhanced backends with write capabilities, completed SafeTensors to ServerlessLLM converter.

Now have complete read-write cycle for all three formats. This was necessary for the converter to work end-to-end, but it's also when the project clearly became more than hypothesis validation. Writing a full format converter with comprehensive error handling is a production system concern, not a research prototype.

The error infrastructure (readers/error.rs, writers/error.rs, traits.rs) was driven by actually using the code. Early versions just unwrapped() everywhere and panicked on errors. Once I started running conversions on real models, I needed proper error reporting to debug issues.

## 2025-11-10

Reorganized project into modular structure: backends/ (format-agnostic I/O), readers/ (format-specific readers), writers/ (format-specific writers), converters/ (high-level orchestration). Implemented mmap loader using OS virtual memory for efficient file access. Created SafeTensors to ServerlessLLM conversion plan.

This restructuring came from the codebase becoming hard to navigate. Earlier structure had everything in loaders/ and it was unclear what depended on what. The new architecture made dependencies explicit: converters use readers and writers, readers/writers use backends.

The mmap approach was interesting - instead of reading files into memory, you map them directly into the address space and let the OS handle paging. For large files this can be more efficient than explicit I/O. Turned out to be a good learning exercise even though it's not always faster in practice.

## 2025-10-30

Created writers module structure mirroring the readers pattern. Analyzed ServerlessLLM format (JSON index + multiple partitioned binary files). Planned 4-phase implementation for conversion.

At this point I was committing to building a real system, not just a benchmark. The decision to plan out converter infrastructure meant I was going well beyond Phase 2. Looking back, this is where I implicitly decided to proceed with Phase 4-style work without formally evaluating whether the hypothesis was proven.

## 2025-10-29

Implemented zero-copy parallel loading for async_io and io_uring. Key idea: pre-allocate single final buffer, split into non-overlapping slices for parallel tasks. Created BufferSlice and BorrowedVec abstractions for safe slice passing to async tasks. Achieved ~50% speedup by eliminating 523MB of memory copies. Added load_range() for efficient byte-range loading.

This was a significant technical breakthrough. The naive parallel approach was: spawn N tasks, each allocates buffer, reads chunk, then copy all chunks into final buffer. That's wasteful - you're allocating and copying 523MB of data unnecessarily.

The zero-copy approach: allocate the final buffer once upfront, split it into N non-overlapping slices, pass each slice to a task, let tasks write directly into the final buffer. No copying needed.

The tricky part was making this safe. Rust's ownership system doesn't like multiple tasks holding mutable references to parts of the same buffer. The BufferSlice abstraction wraps unsafe pointer manipulation behind a safe API that guarantees non-overlapping access.

50% speedup from eliminating memory copies taught me that I/O performance isn't just about the I/O itself - memory operations can dominate. This was more insight than the original io_uring vs tokio comparison.

---

### Reflection: Zero-Copy Innovation

This optimization wasn't in the original plan. The plan was "benchmark io_uring vs tokio::fs" - simple comparison. But actually implementing parallel loading revealed that the bottleneck wasn't the I/O interface at all, it was memory copying.

This is the nature of systems research: you start with a hypothesis about one thing (io_uring syscall overhead) and discover the real problem is something else (memory allocation and copying patterns). The 50% speedup from zero-copy was more significant than any io_uring vs tokio difference.

The BufferSlice abstraction also taught me about safe abstraction over unsafe code. The underlying pointer manipulation is dangerous, but the API makes it impossible to misuse. This is what Rust is good at - unsafe internals, safe surface.

## 2025-10-19

Replaced custom PinnedPool with external zeropool crate. Made tokio loader available on all platforms (previously Linux-only with io_uring). Added comprehensive benchmark suite comparing pooled vs non-pooled variants across sync, io_uring, and tokio backends.

Key lesson: don't reinvent the wheel. I'd spent time building a custom buffer pool, but an external crate already solved this problem better. The zeropool crate was battle-tested and had better performance characteristics than my initial attempt.

Making tokio loader cross-platform was important for testing on non-Linux systems. The original plan assumed Linux-only development, but having portable fallbacks made the code more robust.

## 2025-10-18

Implemented high-performance buffer pool achieving 70% speedup over no-pool baseline. Custom pool beats io_uring without pool: 176ms → 52ms (3.36x faster).

Key optimizations: thread-local storage for lock-free fast path, parking_lot::Mutex instead of std::sync::Mutex, first-fit O(1) allocation instead of best-fit O(n), unsafe set_len() to avoid zero-filling 500MB buffers, data-driven defaults (1MB min buffer, 16 buffer max pool size).

The unsafe set_len() decision was interesting. Safe Vec::resize() was causing massive slowdowns because it zero-fills the buffer. For a 500MB allocation, that's 500MB of unnecessary memory writes. Using unsafe set_len() skips the zeroing since we're about to overwrite the buffer with I/O data anyway.

Initial 4KB min buffer was 130,000x too small for 523MB tensor files. This taught me that "magic values" in code should be data-driven. I benchmarked actual workloads and discovered realistic buffer sizes.

The first-fit vs best-fit lesson was humbling. I implemented best-fit O(n) search thinking it would be "correct" for memory efficiency. Benchmarks showed first-fit O(1) was faster and performed just as well for uniform workloads like tensor loading. Premature optimization toward memory efficiency hurt performance.

---

### Reflection: The Buffer Pool Journey

This was a three-stage evolution: custom implementation → external crate → optimized custom implementation.

Stage 1 (Oct 18): Built custom buffer pool with thread-local storage, discovered unsafe code was necessary for performance, learned that magic values should be empirical not guessed (4KB → 1MB).

Stage 2 (Oct 19): Replaced with zeropool crate, realized I was reinventing the wheel.

Stage 3 (implied later): The final system uses lessons from both - sometimes external crates are better, sometimes custom solutions are needed for specific requirements.

The key insight is that optimization is iterative. You don't know what matters until you measure. Best-fit sounds correct, but first-fit is faster for this workload. Zero-filling sounds safe, but it's unnecessary overhead when you're about to overwrite the memory anyway.

This also taught me about Rust's safety tradeoffs. The unsafe set_len() is genuinely unsafe - if you don't actually fill the buffer afterward, you've got uninitialized memory. But in this controlled context (immediately followed by I/O that fills the buffer), the unsafe code is justified and makes a measurable difference.

## 2025-10-09

Setup io_uring loaders and basic tokio loaders, beat sync loading by 5%.

This was the core hypothesis test. The result: 5% improvement, not 20%. By the original plan's criteria, this should have been a go/no-go decision point. Phase 4 was explicitly conditional on achieving >20% improvement.

I proceeded anyway. Looking back, this was the pivotal moment where the project shifted from hypothesis validation to implementation exploration. The 5% result was interesting enough to keep investigating, and I wanted to understand why the improvement was smaller than expected.

In retrospect, this decision to continue was both good and bad. Good: I discovered valuable insights about zero-copy optimization, buffer pooling, and memory allocation patterns that were more significant than the io_uring vs tokio question. Bad: the project scope expanded well beyond the 300-hour constraint, and I never formally documented the decision to proceed without meeting the stated threshold.

The honest answer is that I found the implementation work engaging and wanted to keep building. The research justification came later.

---

### Reflection: Validating the Core Hypothesis

This is where I should have stopped and written the analysis report. The original Phase 2 deliverable was "benchmark showing performance difference (+  or -)". I had that. 5% improvement, not 20%.

According to the plan, Phase 3 should have been deep analysis of why the improvement was smaller than expected, followed by a go/no-go recommendation. I skipped this entirely and went straight into building production features.

Why did I proceed? A few reasons:

1. The 5% result was tantalizing enough to want to understand better. Was it measurement error? Was there optimization headroom?

2. I was learning a lot from the implementation itself. The buffer pool work, zero-copy optimization, O_DIRECT support - these were valuable engineering lessons independent of the io_uring hypothesis.

3. Honestly, building systems is more fun than writing analysis reports. The implementation pulled me forward.

But this violated the spirit of the original plan. The 300-hour constraint and phased structure were designed to prevent exactly this kind of scope creep. The go/no-go decision point was meant to keep the project focused.

What should I have done? Written the Phase 3 analysis first. Documented that io_uring alone doesn't provide 20% improvement for this workload. Analyzed why (perhaps syscall overhead isn't the bottleneck, perhaps modern tokio is already well-optimized). Then explicitly decided whether to proceed with Phase 4 despite not meeting the threshold, and why.

The lesson: hypothesis-driven research requires discipline to actually evaluate the hypothesis before proceeding. Otherwise it's just feature development dressed up as research.

## 2025-09-30

Scaffolded initial Zig project, then immediately migrated to Rust due to WriterGate breaking changes in Zig I/O system. Completed comprehensive research phase covering ServerlessLLM architecture analysis, io_uring ecosystem research (found tokio-uring crate since Tokio has no built-in support), designed TensorStore format specification with 64-byte alignment for GPU compatibility.

Updated roadmap to focus on validation rather than production system. Established Week 6 go/no-go decision point. Clarified that custom TensorStore format is for learning, real innovation is the io_uring loading approach.

The Zig to Rust pivot happened fast - within a single day. Zig's WriterGate changes made the I/O system unstable for what I needed. Rust had mature io_uring support through tokio-uring and a stable ecosystem.

This taught me about technology selection: stability matters for research projects. Zig is interesting but its rapid evolution meant I'd be fighting the language instead of solving the research problem. Rust's stability let me focus on the actual work.

---

### Reflection: The Zig-to-Rust Pivot

This early decision saved significant time. Had I continued with Zig, I would have been dealing with language ecosystem issues instead of exploring io_uring performance.

The decision factors were:
- Ecosystem maturity: Rust had tokio-uring ready to use. Zig would require more manual io_uring work.
- Stability: Rust's 6-week release cycle with backward compatibility vs Zig's breaking changes.
- Time constraint: 300 hours doesn't leave room for fighting tooling.

In retrospect, starting with Zig was a mistake. I should have evaluated these factors before beginning. But the pivot was quick and the sunk cost was low (one day of scaffolding).

The research phase itself was valuable. Understanding ServerlessLLM's architecture (the OSDI '24 paper) gave me context for what I was trying to improve. Discovering that io_uring support isn't built into Tokio was important - it meant using tokio-uring, which is less mature and has its own quirks.

