# Testing Progress for tensor_store

## Overview

**Goal**: Achieve 80%+ test coverage across all modules
**Current Status**: Phase 1 Complete

---

## Phase 1: Foundation - COMPLETED

**Timeline**: Week 1
**Effort**: ~25 hours estimated
**Status**: Complete

### Accomplishments

#### 1. Test Infrastructure Setup
- Added test dependencies to `Cargo.toml`:
  - `proptest = "1.5"` - Property-based testing
  - `tempfile = "3.14"` - Temporary file management
  - `tokio-test = "0.4"` - Tokio testing utilities

#### 2. Enhanced Existing Tests
- **src/backends/odirect.rs** (6 → 15 tests, +9 tests)
  - `test_align_to_block_large_sizes` - Large size alignment
  - `test_is_block_aligned_edge_cases` - Alignment edge cases
  - `test_can_use_direct_read_aligned` - Direct read validation
  - `test_can_use_direct_read_unaligned` - Unaligned read validation
  - `test_can_use_direct_write_edge_cases` - Write validation
  - `test_pad_to_block` - Padding correctness
  - `test_alloc_aligned_large` - Large buffer allocation
  - `test_owned_aligned_buffer_into_vec_exceeds_capacity` - Capacity overflow
  - `test_aligned_chunk_out_of_bounds` - Bounds checking

- **src/backends/buffer_slice.rs** (3 → 8 tests, +5 tests)
  - `test_buffer_slice_large_slice` - 1MB slice handling
  - `test_buffer_slice_full_array` - Full array operations
  - `test_buffer_slice_single_element` - Single element slice
  - `test_buffer_slice_non_overlapping` - Non-overlapping verification
  - `test_buffer_slice_send` - Send trait verification

#### 3. New Module Tests
- **src/backends/mmap.rs** (0 → 12 tests)
  - `test_mmap_basic` - Basic mmap operations
  - `test_mmap_empty_file_error` - Empty file handling
  - `test_mmap_large_file` - 10MB file mapping
  - `test_mmap_range_aligned` - Aligned range mapping
  - `test_mmap_range_unaligned` - Unaligned range mapping
  - `test_mmap_range_page_boundary` - Page boundary handling
  - `test_mmap_range_exceeds_file` - Range validation
  - `test_mmap_range_zero_length_error` - Zero-length error
  - `test_mmap_clone` - Clone operation
  - `test_mmap_deref` - Deref trait
  - `test_mmap_as_ref` - AsRef trait
  - `test_mmap_range_full_file` - Full file range mapping

- **src/backends/batch.rs** (0 → 6 tests)
  - `test_group_requests_by_file_single_file` - Single file grouping
  - `test_group_requests_by_file_multiple_files` - Multi-file grouping
  - `test_group_requests_by_file_empty` - Empty request handling
  - `test_flatten_results_all_success` - Result flattening
  - `test_flatten_results_preserves_order` - Order preservation
  - `test_flatten_results_empty` - Empty results

- **src/backends/mod.rs** (0 → 5 tests)
  - `test_buffer_pool_initialization` - Pool initialization
  - `test_buffer_pool_singleton` - Singleton pattern
  - `test_buffer_pool_config_values` - Config validation
  - `test_get_buffer_pool_multiple_calls` - Multiple call stability
  - `test_platform_specific_exports` - Platform-specific compilation

#### 4. GitLab CI Enhancements
- Updated `test-unit` job with `--show-output` flag
- Enhanced `coverage-llvm` job:
  - Added `cargo llvm-cov clean --workspace` before run
  - Changed to `--all-targets` for comprehensive coverage
  - Added HTML report generation
  - Improved coverage percentage extraction
  - Added HTML artifacts to output

### Results

**Total Tests**: 46 (up from 9)
**All Tests**: PASSING
**Test Time**: ~0.06s

| Module | Before | After | Added |
|--------|--------|-------|-------|
| odirect.rs | 6 | 15 | +9 |
| buffer_slice.rs | 3 | 8 | +5 |
| mmap.rs | 0 | 12 | +12 |
| batch.rs | 0 | 6 | +6 |
| mod.rs | 0 | 5 | +5 |
| **TOTAL** | **9** | **46** | **+37** |

---

## Phase 2: I/O Backends - TODO

**Timeline**: Week 2
**Effort**: ~30 hours
**Goal**: Backend modules at 80%+ coverage
**Status**: Pending

### Modules to Test

#### 1. io_uring.rs (470 lines) - 12 hours
**Current Coverage**: 0%
**Target Coverage**: 85%
**Tests to Add**: ~25 tests

**Unit Tests to Add**:
```rust
// Validation helpers
test_validate_read_count_exact
test_validate_read_count_short_read
test_validate_chunk_size_within_limits
test_validate_chunk_size_exceeds_isize_max

// Chunk request building
test_build_chunk_requests_single_chunk
test_build_chunk_requests_multiple_chunks
test_build_chunk_requests_uneven_division
test_build_chunk_requests_alignment_padding

// File size helpers
test_statx_file_size_normal
test_statx_file_size_large

// Fallback detection
test_allow_direct_fallback_einval
test_allow_direct_fallback_eopnotsupp
test_allow_direct_fallback_other_errors

// Core operations (tokio-uring runtime)
test_load_small_file
test_load_empty_file
test_load_parallel_single_chunk
test_load_parallel_multiple_chunks
test_load_parallel_with_alignment
test_load_range_aligned
test_load_range_unaligned
test_write_all_direct
test_write_all_fallback
test_batch_load_multiple_files
```

**Notes**:
- All tests must use `#[cfg(target_os = "linux")]`
- Use `tokio::test(flavor = "current_thread")` for async tests
- Test with tempfile for I/O operations
- Test O_DIRECT fallback scenarios

#### 2. async_io.rs (514 lines) - 10 hours
**Current Coverage**: 0%
**Target Coverage**: 85%
**Tests to Add**: ~20 tests

**Unit Tests to Add**:
```rust
// Helper functions
test_div_ceil_basic
test_div_ceil_exact_division
test_div_ceil_zero

// Linux-specific O_DIRECT path
#[cfg(target_os = "linux")]
test_load_with_direct_io
test_load_fallback_to_buffered
test_write_all_direct_aligned
test_write_all_direct_unaligned
test_load_parallel_direct

// Cross-platform tests
test_load_small_file
test_load_empty_file
test_load_range_basic
test_load_range_eof
test_write_all_basic
test_write_all_empty
test_load_parallel_vs_sequential
```

**Notes**:
- Cross-platform compatibility required
- Test both O_DIRECT (Linux) and buffered I/O paths
- Use `#[tokio::test]` for async tests
- Verify parallel vs sequential equivalence

#### 3. sync_io.rs (491 lines) - 8 hours
**Current Coverage**: 0%
**Target Coverage**: 80%
**Tests to Add**: ~15 tests

**Unit Tests to Add**:
```rust
// Blocking operations
test_load_small_file
test_load_empty_file
test_load_range_basic
test_load_range_eof
test_write_all_basic
test_write_all_empty

// Platform-specific
#[cfg(target_os = "linux")]
test_load_parallel_threaded

// Batch operations
test_batch_load_sync
test_load_range_batch_single
test_load_range_batch_multiple
```

**Notes**:
- Blocking tests (no async)
- Test thread-based parallelism
- Verify compatibility with sync contexts

---

## Phase 3: Readers - TODO

**Timeline**: Week 3
**Effort**: ~26 hours
**Goal**: Reader modules at 80%+ coverage
**Status**: Pending

### Modules to Test

#### 1. serverlessllm.rs (1,357 lines) - 16 hours
**Current Coverage**: 0%
**Target Coverage**: 80%
**Tests to Add**: ~30 tests

**Unit Tests to Add**:
```rust
// ServerlessLLMIndex tests
test_index_new
test_index_from_json_valid
test_index_from_json_invalid
test_index_tensor_exists
test_index_tensor_not_found
test_index_tensor_names
test_index_tensor_count
test_index_clone

// Partition validation
test_validate_partition_offset_valid
test_validate_partition_offset_exceeds_size
test_validate_partition_offset_overflow

// Tensor/TensorMmap tests
test_tensor_metadata_accessors
test_tensor_data_integrity

// ServerlessLLM/ServerlessLLMMmap tests
test_serverlessllm_new_empty
test_serverlessllm_tensor_lookup
test_serverlessllm_iter

// Async operations
test_parse_index_valid_file
test_parse_index_missing_file
test_load_tensor_single
test_load_tensors_batch

// Sync operations
test_parse_index_sync_valid
test_load_sync_basic
test_load_mmap_basic
```

**Integration Tests Needed**:
- Load GPT-2 fixture from `fixtures/openai-community-gpt2/`
- Verify tensor counts and names
- Test with all available fixtures
- Compare eager vs mmap loading

#### 2. safetensors.rs (313 lines) - 8 hours
**Current Coverage**: 0%
**Target Coverage**: 85%
**Tests to Add**: ~18 tests

**Unit Tests to Add**:
```rust
// SafeTensorsOwned tests
test_owned_from_bytes_valid
test_owned_from_bytes_invalid
test_owned_as_bytes
test_owned_into_bytes
test_owned_tensors_accessor

// SafeTensorsMmap tests
test_mmap_from_file
test_mmap_tensors_accessor
test_mmap_clone

// Async operations
test_load_async_basic
test_load_parallel_chunks

// Sync operations
test_load_sync_basic
test_load_mmap_basic

// Trait implementations
test_async_reader_trait_load_tensor
test_sync_reader_trait_tensor_metadata
```

**Integration Tests Needed**:
- Load all SafeTensors fixtures
- Verify mmap vs owned equivalence
- Test with real model files (GPT-2, Qwen2, etc.)

#### 3. Other Reader Modules (error.rs, traits.rs, mod.rs) - 2 hours
**Current Coverage**: 0%
**Target Coverage**: 60%
**Tests to Add**: ~8 tests

**Tests to Add**:
- Error variant construction
- Error Display implementation
- Error From conversions
- Trait default implementations

---

## Phase 4: Writers & Converters - TODO

**Timeline**: Week 4
**Effort**: ~20 hours
**Goal**: Writers/converters at 85%+ coverage
**Status**: Pending

### Modules to Test

#### 1. serverlessllm.rs writer (273 lines) - 6 hours
**Current Coverage**: 0%
**Target Coverage**: 85%
**Tests to Add**: ~12 tests

**Unit Tests to Add**:
```rust
test_writer_new
test_write_index_basic
test_write_partition_basic
test_write_empty_index
test_write_index_sync
test_write_partition_sync
```

**Integration Tests Needed**:
- Write tensors, read back, verify equality
- Test with large partitions (multi-GB)
- Roundtrip with fixtures

#### 2. safetensors.rs writer (169 lines) - 4 hours
**Current Coverage**: 0%
**Target Coverage**: 85%
**Tests to Add**: ~10 tests

**Unit Tests to Add**:
```rust
test_writer_new
test_write_basic
test_write_sync
test_write_empty
```

**Integration Tests Needed**:
- Write/read roundtrip
- Verify format compatibility

#### 3. safetensors_to_serverlessllm.rs (198 lines) - 8 hours
**Current Coverage**: 0%
**Target Coverage**: 90%
**Tests to Add**: ~15 tests

**Unit Tests to Add**:
```rust
// Helper functions
test_calculate_contiguous_stride
test_dtype_to_serverlessllm_all_types
test_dtype_to_serverlessllm_unsupported

// Partitioning
test_greedy_partition_empty
test_greedy_partition_single_partition
test_greedy_partition_multiple_partitions
test_greedy_partition_balanced

// Conversion
test_convert_basic
test_convert_invalid_partition_count
```

**Integration Tests Needed**:
- Convert all fixtures SafeTensors → ServerlessLLM
- Verify data integrity (load both formats, compare tensors)
- Test with various partition counts (1, 2, 4, 8)

#### 4. types/ modules - 2 hours
**Current Coverage**: 0%
**Target Coverage**: 95%
**Tests to Add**: ~8 tests

**Unit Tests to Add**:
```rust
test_tensor_entry_default
test_tensor_entry_clone
test_tensor_entry_partial_eq
test_tensor_entry_serialize
test_tensor_entry_deserialize
test_tensor_entry_roundtrip
```

---

## Phase 5: Property Tests & Integration - TODO

**Timeline**: Week 5
**Effort**: ~16 hours
**Goal**: Property tests + comprehensive integration
**Status**: Pending

### Property-Based Tests

#### 1. Serialization Properties (4 hours)
**File**: `src/types/serverlessllm.rs` (inline with #[cfg(test)])

```rust
use proptest::prelude::*;

prop_compose! {
    fn arbitrary_tensor_entry()(
        offset in 0u64..1_000_000,
        size in 1u64..1_000_000,
        shape in prop::collection::vec(0i64..1000, 1..5),
        dtype in prop::sample::select(vec!["float32", "int64", "bfloat16"]),
        partition_id in 0usize..100,
    ) -> TensorEntry {
        TensorEntry { offset, size, shape, stride: vec![], dtype, partition_id }
    }
}

proptest! {
    #[test]
    fn prop_tensor_entry_json_roundtrip(entry in arbitrary_tensor_entry()) {
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: TensorEntry = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(entry, parsed);
    }

    #[test]
    fn prop_index_roundtrip(
        tensors in prop::collection::hash_map(
            "[a-z]{1,10}",
            arbitrary_tensor_entry(),
            0..20
        )
    ) {
        // Serialize index, parse back, verify equality
    }
}
```

#### 2. Alignment Properties (4 hours)
**File**: `src/backends/odirect.rs` (inline with #[cfg(test)])

```rust
proptest! {
    #[test]
    fn prop_align_to_block_is_aligned(size in 0usize..1_000_000) {
        let aligned = align_to_block(size);
        prop_assert_eq!(aligned % BLOCK_SIZE, 0);
        prop_assert!(aligned >= size);
        if size > 0 {
            prop_assert!(aligned - size < BLOCK_SIZE);
        }
    }

    #[test]
    fn prop_chunk_coverage(
        file_size in 1usize..100_000_000,
        chunks in 1usize..256
    ) {
        // Verify chunks cover entire file without gaps/overlaps
    }
}
```

#### 3. Data Integrity Properties (4 hours)
**File**: `src/backends/async_io.rs` or `io_uring.rs` (inline)

```rust
proptest! {
    #[test]
    fn prop_write_read_preserves_data(data in prop::collection::vec(any::<u8>(), 0..10_000)) {
        tokio_test::block_on(async {
            let tmpfile = NamedTempFile::new().unwrap();

            write_all(tmpfile.path(), data.clone()).await.unwrap();
            let read_data = load(tmpfile.path()).await.unwrap();

            prop_assert_eq!(data, read_data);
        });
    }

    #[test]
    fn prop_parallel_equals_sequential(
        data in prop::collection::vec(any::<u8>(), 1000..100_000),
        chunks in 2usize..32
    ) {
        // Verify parallel loading matches sequential
    }
}
```

#### 4. End-to-End Integration Tests (4 hours)
**File**: `src/converters/safetensors_to_serverlessllm.rs` (inline)

```rust
#[tokio::test]
async fn test_conversion_roundtrip_gpt2() {
    use tempfile::TempDir;

    let tmpdir = TempDir::new().unwrap();

    // Convert SafeTensors → ServerlessLLM
    convert_safetensors_to_serverlessllm(
        "fixtures/openai-community-gpt2/model.safetensors",
        tmpdir.path().to_str().unwrap(),
        4,
    ).await.unwrap();

    // Load both formats
    let safetensors = safetensors::load("fixtures/openai-community-gpt2/model.safetensors").await.unwrap();
    let serverlessllm = serverlessllm::load(tmpdir.path()).await.unwrap();

    // Verify tensor counts match
    assert_eq!(safetensors.tensor_names().len(), serverlessllm.tensor_names().len());

    // Verify tensor data matches
    for name in safetensors.tensor_names() {
        let st_tensor = safetensors.tensor(name).unwrap();
        let sllm_tensor = serverlessllm.tensor(name).unwrap();
        assert_eq!(st_tensor.data(), sllm_tensor.data());
    }
}
```

---

## Phase 6: Verification & Polish - TODO

**Timeline**: Week 6
**Effort**: ~8 hours
**Goal**: Verify 80%+ coverage, all CI green
**Status**: Pending

### Tasks

#### 1. Coverage Analysis (2 hours)
- Run `cargo llvm-cov --workspace --all-targets`
- Generate HTML reports
- Identify gaps in coverage
- Document untestable code (if any)

**Commands**:
```bash
cd tensor_store
cargo llvm-cov clean --workspace
cargo llvm-cov --workspace --all-targets --html
cargo llvm-cov report --summary-only
```

**Target Output**:
```
Filename                      Regions    Missed Regions     Cover   Functions  Missed Functions  Executed
-----------------------------------------------------------------------------------------------------------------
src/backends/async_io.rs          245                42    82.86%          18                 2    88.89%
src/backends/batch.rs              18                 0   100.00%           2                 0   100.00%
src/backends/buffer_slice.rs       12                 0   100.00%           3                 0   100.00%
src/backends/io_uring.rs          298                45    84.90%          21                 3    85.71%
src/backends/mmap.rs               28                 2    92.86%           2                 0   100.00%
src/backends/mod.rs                15                 3    80.00%           5                 1    80.00%
src/backends/odirect.rs            87                 8    90.80%          11                 1    90.91%
src/backends/sync_io.rs           189                38    79.89%          15                 3    80.00%
src/converters/...                142                14    90.14%          12                 1    91.67%
src/readers/...                   487                97    80.08%          38                 8    78.95%
src/writers/...                   156                31    80.13%          14                 3    78.57%
src/types/...                      12                 0   100.00%           8                 0   100.00%
-----------------------------------------------------------------------------------------------------------------
TOTAL                            1689               280    83.42%         149                22    85.23%
```

#### 2. Fill Coverage Gaps (4 hours)
- Add tests for uncovered edge cases
- Test error paths not yet covered
- Add tests for unsafe blocks where possible
- Document why certain code cannot be tested (e.g., OS-specific unsafe code)

**Gap Analysis Process**:
1. Review `target/llvm-cov/html/index.html`
2. Identify files with <80% coverage
3. Click into files to see uncovered lines (highlighted in red)
4. Write targeted tests for uncovered branches

**Common Gap Areas**:
- Error handling paths (file not found, permission denied, etc.)
- Platform-specific code paths
- Unsafe blocks (may need integration tests)
- Initialization code (may run only once)

#### 3. CI Validation (2 hours)
- Push changes to GitLab
- Verify all CI jobs pass:
  - ✅ format-check
  - ✅ clippy
  - ✅ build-release
  - ✅ test-unit
  - ✅ test-doc
  - ✅ coverage-llvm
  - ✅ bench-compile
  - ✅ docs-build
- Check coverage percentage in CI logs
- Verify artifacts are generated (coverage.lcov, HTML reports)

**Expected CI Output**:
```
Total coverage: 83.42 percent
```

---

## Success Criteria Checklist

### Primary Metrics
- [ ] Total code coverage >= 80%
- [ ] Test count >= 240
- [ ] All CI jobs passing

### Module-Specific Coverage
- [ ] backends/odirect.rs >= 90%
- [ ] backends/buffer_slice.rs >= 95%
- [ ] backends/io_uring.rs >= 85%
- [ ] backends/async_io.rs >= 85%
- [ ] backends/sync_io.rs >= 80%
- [ ] backends/mmap.rs >= 90%
- [ ] backends/batch.rs >= 95%
- [ ] backends/mod.rs >= 70%
- [ ] readers/serverlessllm.rs >= 80%
- [ ] readers/safetensors.rs >= 85%
- [ ] writers/serverlessllm.rs >= 85%
- [ ] writers/safetensors.rs >= 85%
- [ ] converters/safetensors_to_serverlessllm.rs >= 90%
- [ ] types/* >= 95%

### Quality Indicators
- [ ] All public APIs tested
- [ ] All error variants triggered in tests
- [ ] All `#[cfg]` branches tested (Linux vs non-Linux)
- [ ] Integration tests use real fixtures
- [ ] Property tests verify critical invariants
- [ ] Zero flaky tests
- [ ] Platform coverage: Linux + at least one non-Linux platform

### Infrastructure
- [ ] GitLab CI enhanced with coverage reporting
- [ ] Coverage artifacts published (lcov + HTML)
- [ ] Property-based tests integrated
- [ ] Fixture-based integration tests working

---

## Test Execution Guide

### Run All Tests
```bash
cd tensor_store
cargo test --workspace --lib --bins
```

### Run Specific Module Tests
```bash
# Backends
cargo test --lib backends::odirect::tests
cargo test --lib backends::mmap::tests
cargo test --lib backends::batch::tests

# Readers (when added)
cargo test --lib readers::serverlessllm::tests
cargo test --lib readers::safetensors::tests

# Writers (when added)
cargo test --lib writers::serverlessllm::tests
```

### Run with Coverage
```bash
cargo llvm-cov --workspace --all-targets --html
cargo llvm-cov report --summary-only
# Open target/llvm-cov/html/index.html in browser
```

### Run Property Tests
```bash
# Run all property tests
cargo test --release -- proptest

# Run specific property test
cargo test --release -- prop_tensor_entry_json_roundtrip
```

### Run Tests on Specific Platform
```bash
# Linux-specific tests
cargo test --lib -- io_uring
cargo test --lib -- odirect

# Cross-platform tests
cargo test --lib -- async_io
```

---

## Notes and Reminders

### Test Patterns to Follow

1. **Use tempfile for file operations**:
   ```rust
   use tempfile::NamedTempFile;

   let mut tmpfile = NamedTempFile::new().unwrap();
   tmpfile.write_all(b"test data").unwrap();
   tmpfile.flush().unwrap();
   ```

2. **Use fixtures for integration tests**:
   ```rust
   const GPT2_SAFETENSORS: &str = "fixtures/openai-community-gpt2/model.safetensors";

   #[tokio::test]
   async fn test_load_gpt2() {
       let data = load(GPT2_SAFETENSORS).await.unwrap();
       assert!(data.len() > 0);
   }
   ```

3. **Platform-specific tests**:
   ```rust
   #[cfg(target_os = "linux")]
   #[tokio::test]
   async fn test_io_uring_specific() {
       // io_uring-specific test
   }

   #[cfg(not(target_os = "linux"))]
   #[tokio::test]
   async fn test_async_io_fallback() {
       // async_io fallback test
   }
   ```

4. **Property-based tests**:
   ```rust
   use proptest::prelude::*;

   proptest! {
       #[test]
       fn prop_name(input in strategy) {
           // Property to verify
       }
   }
   ```

### Common Pitfalls to Avoid

1. **Don't test implementation details** - Test behavior, not internals
2. **Don't create fixtures in tests** - Use tempfile for temp data, fixtures/ for real data
3. **Don't skip error paths** - Test failure cases explicitly
4. **Don't use hardcoded paths** - Use `tempfile` or `fixtures/` constants
5. **Don't ignore platform differences** - Use `#[cfg]` appropriately

### Good Testing Practices

1. **Test one thing per test** - Keep tests focused
2. **Use descriptive test names** - `test_load_empty_file_error` not `test_load_2`
3. **Test edge cases** - Empty, zero, max values, boundaries
4. **Clean up resources** - tempfile does this automatically
5. **Use assertions effectively** - `assert_eq!` with helpful messages

---

## Current Status Summary

**Phase 1**: Complete (46 tests, all passing)
**Phase 2**: Pending (io_uring, async_io, sync_io)
**Phase 3**: Pending (readers)
**Phase 4**: Pending (writers, converters)
**Phase 5**: Pending (property tests, integration)
**Phase 6**: Pending (verification, polish)

**Next Steps**:
1. Start Phase 2 with `io_uring.rs` tests
2. Follow with `async_io.rs` and `sync_io.rs`
3. Continue through remaining phases

**Estimated Time to Completion**: ~99 hours remaining (5 weeks)
