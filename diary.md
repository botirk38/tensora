# Project Development Diary

## 2026-03-08

Refactored Python bindings to align with new architecture. Deleted the `bridge/` module which was unnecessarily normalizing tensor data. Renamed `torch/` to `convert/` with a `TensorData` struct to reduce function arguments.

Split the `api/` module by format instead of function type - created `api/safetensors.rs` and `api/serverlessllm.rs` instead of `handles.rs` + `functions.rs`. This matches the core crate's reorganization.

Fixed several bugs discovered during the refactor:
- `SafeTensorsMmap::from_owned()` was calling non-existent `from_bytes()` - removed the broken method
- ServerlessLLM was using `.keys()` instead of `.tensor_names()`, `.open_mmap()` instead of module-level `load_mmap()`
- ServerlessLLM `.tensor()` returns `Option` not `Result` - added `tensor_not_found()` helper

Standardized API between formats:
- `open_*_sync` now uses `load_sync()` → owned in-memory
- `open_*_mmap` now uses `load_mmap()` → file-backed mmap
- Both use inner enum to handle Owned vs Mmap variants

Fixed clippy warnings: saturating_mul, redundant closures, too_many_arguments (via TensorData struct).

All 73 Python tests pass, clippy clean on both core and bindings.

Inlined helper modules into their appropriate places:
- `validate_path_exists` into safetensors.rs and serverlessllm.rs
- `load_safetensors_async` and `load_serverlessllm_async` into their respective modules
- Merged `dtype_map.rs` into `torch.rs`
- Deleted validation.rs, runtime_async.rs, and dtype_map.rs to reduce abstraction layers

Fixed clippy: PathBuf → Path, removed unused imports.

## 2025-02-11

Code review day that turned into comprehensive refactoring. Started with clippy warnings, ended up touching 15+ files.

Fixed clippy pedantic warnings - unnested or-patterns, redundant closures, underscores in large literals. Also converted ReaderError to use thiserror (was manual Display/Error impls copy-pasted from WriterError). Added rust-version = "1.92" to Cargo.toml.

Bigger architectural change: refactored trait signatures from `impl AsRef<Path>` to `&Path`. The impl version looks nicer but breaks object safety - can't make trait objects. `&Path` is more restrictive but enables dynamic dispatch. Callers just add `.as_ref()` at call sites.

Error handling overhaul was the real work. Added 8 rich error variants to `ReaderError`:
- `TensorNotFound { name }` - better than generic "not found" string
- `PartitionNotFound { partition_id, path }` - tells you exactly which partition failed
- `PartitionTooSmall { path, actual, required }` - actionable: shows the size mismatch
- `OffsetOverflow { name }`, `SizeTooLarge { name, size }` - safer than panicking
- `MutexPoisoned` - thread panic handling
- Plus `InvalidIndexFormat`, `JsonParseError` for parsing issues

Then replaced ~15 string-based errors in serverlessllm/reader.rs with the new variants. Replaced `.unwrap()` calls on mutex locks with proper error handling. Fixed HashMap lookups that assumed keys existed.

Documentation pass: added `# Errors` sections to ~50 public functions across all backends and both reader modules. Each function now documents what can go wrong.

Final cleanup: replaced `use super::*` with explicit imports in 29 test modules. Wildcard imports hide what you're actually using. The io_uring tests were tricky - nested modules with different imports.

Spent time improving unsafe documentation too. The BufferSlice and odirect code does tricky pointer stuff for zero-copy I/O. Added proper SAFETY comments explaining preconditions and invariants.

8 commits, all 173 tests passing, clippy clean, docs generate without warnings.

This wasn't planned. Started with "fix a few clippy warnings" and scope crept. But it's the right kind of scope creep - technical debt that needed addressing. The error variants will make debugging easier. When something fails, users get specific context instead of generic strings.

## 2025-12-17

Documentation week. The project had accumulated a lot of doc debt - broken links in the main README, empty files, no module docs. Spent time fixing all of it.

Created READMEs for every major module (backends, safetensors, serverlessllm, benches). Added a proper CHANGELOG. Updated the main README with a Quick Demo section so people can actually try the project without reading everything first.

The bigger win was fixing the fixture download workflow. Before, users had to remember to pass `--convert` to get ServerlessLLM format, and if they forgot, demos would fail with confusing errors. Changed it so conversion happens by default. Also added a heuristic that picks partition count based on model size - small models get 4 partitions, large ones get up to 32. Users shouldn't need to understand partition strategies to get reasonable performance.

The script now auto-builds the converter binary if it's missing. Small thing, but eliminates a whole class of "binary not found" errors.

Writing docs forced me to actually try the demo workflow from scratch. That's when I realized how clunky it was. Documentation isn't just explaining - it's testing your UX assumptions.

## 2025-12-08

Reorganized the entire project structure from a flat layout into format-based organization. Previously everything was mixed together - now there's clear separation:
- `src/safetensors/` - reader.rs, writer.rs, mod.rs
- `src/serverlessllm/` - reader.rs, writer.rs, types.rs, mod.rs
- `src/backends/` - all I/O implementations
- `src/converters/` - format conversion logic
- `src/types/` - shared traits and errors

This came from the codebase getting hard to navigate. When everything's in one place, you waste time hunting for the right file. The new structure makes it obvious where things live.

Also fixed the load_parallel race condition that was causing intermittent test failures. The issue was shared file descriptors across async tasks - each parallel chunk was stepping on the others. Fixed by giving each task its own file handle.

Added comprehensive tests for both readers and writers modules. Test count is now much higher and covers the actual edge cases I've hit in practice.

## 2025-12-02

Refactored ServerlessLLM code to eliminate duplication. Added more backend tests. CI maintenance - updated dependencies, fixed pipeline steps. Flattened the repo structure.

The refactoring was cleanup - removing unused functions and consolidating duplicate code. Not exciting but necessary.

## 2025-11-29

Added 104+ tests tracked in TESTING_PROGRESS.md. Set up full GitLab CI pipeline - build, test, lint, security checks. Wrote 885-line demo binary for format comparison.

This wasn't planned. After fixing the race condition in load_parallel, I realized I had no way to verify it actually worked. The codebase had grown beyond a simple benchmark - O_DIRECT, parallel loading, buffer pools all have subtle correctness requirements.

The CI made sense after I forgot to run clippy and committed warnings. Automating meant the computer catches what I forget. cargo-audit caught a vulnerable dependency I didn't know about.

The demo binary was for showing my supervisor performance numbers. I kept manually re-running benchmarks. Needed something reproducible with proper warmup and statistics. Cold cache effects matter when claiming performance improvements.

Testing felt overwhelming until I broke it down in TESTING_PROGRESS.md. Phase 1: 46 tests. Phase 2: 72 tests. Writing it down made it manageable.

CI had flaky tests initially from shared state. Fixed with tempfile for proper isolation. Test code needs the same quality as production code.

## 2025-11-25

Implemented O_DIRECT support. Added odirect.rs backend, refactored async_io/io_uring/sync_io to support it, added batch processing module.

O_DIRECT bypasses the page cache. For multi-GB model files you read once, the page cache is overhead. Why cache what you'll never read again?

Implementation required handling alignment constraints - O_DIRECT needs 512-byte aligned buffers and offsets. Wrong alignment gives EINVAL from the kernel. Created abstractions for align_to_block(), alloc_aligned(), OwnedAlignedBuffer.

Added batch processing because O_DIRECT works best with larger I/O ops. Small reads perform poorly. The batch.rs module groups requests efficiently.

This is way beyond the original "minimal validation" plan. At this point I was optimizing for real-world performance, not testing hypotheses. O_DIRECT is production stuff, not research prototype.

## 2025-11-24

Added buffer pool builder with presets (small/medium/large workloads). Implemented profiler binary. Created script to download HuggingFace models and convert to ServerlessLLM.

The builder came from frustration with magic numbers. Earlier I learned my 4KB buffer was 130,000x too small. Wanted an API where good defaults are easy and bad defaults are hard. Presets encode learnings from benchmarks.

Profiler binary for debugging performance regressions. Benchmarks tell you what's fast/slow, profiler shows where CPU time goes.

HuggingFace script was practical necessity. Needed real model files, not synthetic data. Real .safetensors have different characteristics than random bytes.

## 2025-11-23

Decoupled mmap and sync I/O backends into separate modules. Pure refactoring - no new functionality.

The mmap backend was tangled with sync I/O code. Separating them let me optimize each independently. Should have done this earlier.

## 2025-11-22

Parallelized the SafeTensors to ServerlessLLM converter using load_parallel() and futures::try_join_all. Created shared types/ module for TensorEntry.

The converter was painfully slow. Sequential loading then sequential writing scaled linearly with model size. Parallelizing both made it usable for large models.

The types/ refactoring fixed a bug - reader and writer had different TensorEntry definitions that should have been the same. Extracting to shared module prevents drift. DRY isn't just reducing duplication, it's having one source of truth.

## 2025-11-21

Implemented complete ServerlessLLM format support - reader, writer, converter. Added error handling infrastructure. Implemented SafeTensors writer.

Now have complete read-write cycle for all formats. This is when the project clearly became more than hypothesis validation. Full format converter with comprehensive error handling is a production concern, not a research prototype.

The error infrastructure was driven by actually using the code. Early versions just unwrapped() everywhere. Running conversions on real models required proper error reporting to debug issues.

## 2025-11-10

Reorganized into modular structure: backends/ (format-agnostic I/O), readers/ (format-specific), writers/, converters/. Implemented mmap loader. Planned SafeTensors to ServerlessLLM conversion.

Earlier structure had everything in loaders/ - unclear dependencies. New architecture makes it explicit: converters use readers/writers, which use backends.

mmap was interesting - map files into address space, let OS handle paging. Can be more efficient for large files than explicit I/O. Good learning exercise even if not always faster in practice.

## 2025-10-30

Created writers module mirroring readers pattern. Analyzed ServerlessLLM format (JSON index + partitioned binaries). Planned 4-phase conversion implementation.

This is where I committed to building a real system, not just a benchmark. Looking back, implicitly decided to proceed with Phase 4-style work without formally evaluating whether I'd proven the hypothesis.

## 2025-10-29

Implemented zero-copy parallel loading. Pre-allocate final buffer, split into non-overlapping slices for parallel tasks. Created BufferSlice and BorrowedVec abstractions. ~50% speedup by eliminating 523MB of memory copies.

The naive approach: spawn N tasks, each allocates buffer, reads chunk, copy into final buffer. Wasteful.

Zero-copy: allocate final buffer once, split into N slices, pass to tasks, write directly. No copying.

Tricky making it safe. Rust doesn't like multiple tasks with mutable refs to same buffer. BufferSlice wraps unsafe pointer manipulation behind safe API guaranteeing non-overlapping access.

Big insight: I/O performance isn't just about the I/O. Memory operations can dominate. This was more valuable than the io_uring vs tokio comparison.

## 2025-10-19

Replaced custom PinnedPool with zeropool crate. Made tokio loader cross-platform (was Linux-only). Added comprehensive benchmarks comparing pooled vs non-pooled.

Lesson: don't reinvent the wheel. Spent time building custom buffer pool, but external crate was better. zeropool was battle-tested with better performance.

Cross-platform tokio was important for testing on non-Linux systems. Having portable fallbacks made the code more robust.

## 2025-10-18

Implemented high-performance buffer pool. 70% speedup over no-pool baseline. 176ms → 52ms (3.36x faster).

Key optimizations: thread-local storage for lock-free fast path, parking_lot::Mutex, first-fit O(1) allocation instead of best-fit O(n), unsafe set_len() to skip zero-filling 500MB buffers.

The unsafe set_len() was interesting. Vec::resize() zero-fills - for 500MB that's massive overhead. Using unsafe skips zeroing since we're about to fill with I/O anyway.

Initial 4KB min buffer was 130,000x too small for 523MB files. Learned magic values should be data-driven. Benchmarked actual workloads to find realistic sizes.

First-fit vs best-fit was humbling. Implemented best-fit O(n) thinking it was "correct" for memory efficiency. Benchmarks showed first-fit O(1) was faster and worked just as well. Premature optimization hurt performance.

## 2025-10-09

Set up io_uring and tokio loaders. Beat sync loading by 5%.

This was the core hypothesis test. Result: 5% improvement, not 20%. By the plan's criteria, this should have been go/no-go.

I proceeded anyway. This was the pivotal moment where the project shifted from hypothesis validation to implementation exploration. The 5% was tantalizing enough to investigate further.

In retrospect, both good and bad. Good: discovered valuable insights about zero-copy, buffer pooling, memory allocation. Bad: scope expanded beyond 300-hour constraint, never formally documented the decision to proceed.

Honest answer: I found implementation engaging and wanted to keep building. Research justification came later.

Should have written Phase 3 analysis first. Documented why io_uring doesn't provide 20% improvement. Analyzed reasons - maybe syscall overhead isn't the bottleneck, maybe tokio is already well-optimized. Then explicitly decided whether to proceed and why.

Lesson: hypothesis-driven research needs discipline to evaluate the hypothesis before proceeding. Otherwise it's just feature development.

## 2025-09-30

Started with Zig, immediately pivoted to Rust due to WriterGate breaking changes in Zig I/O. Completed research phase - ServerlessLLM analysis, io_uring ecosystem research, TensorStore format design.

Zig pivot happened in one day. Zig's rapid evolution meant fighting the language instead of solving the research problem. Rust had mature io_uring support (tokio-uring) and stable ecosystem.

Lesson: stability matters for research projects. Time constraint doesn't leave room for fighting tooling.

Research phase was valuable. Understanding ServerlessLLM (OSDI '24 paper) gave context. Learning that Tokio doesn't have built-in io_uring support was important - meant using tokio-uring, which is less mature.
