//! Storage engine vocabulary types.
//!
//! This module defines the request/result types used by all storage engines.
//! Engines are introduced in subsequent PRs; this module is the shared vocabulary
//! they all speak.
//!
//! # Request types
//!
//! All requests borrow their path as `&Path` to avoid allocation at the call site.
//! Builders are provided for ergonomics.
//!
//! - [`FileReadRequest`] — read an entire file
//! - [`RangeReadRequest`] — read a byte range from a file
//! - [`BatchReadRequest`] — read multiple ranges, possibly from multiple files
//! - [`WriteAtRequest`] — write bytes at a specific offset in an open file
//! - [`MmapRequest`] — memory-map an entire file
//! - [`MmapRangeRequest`] — memory-map a byte range within a file
//!
//! # Result types
//!
//! - [`RangeReadResult`] — single result from a batch read

pub mod availability;
pub mod buffer;

pub use std::io::Result as IoResult;

use std::path::Path;

// ============================================================================
// FileReadRequest
// ============================================================================

/// Request to read an entire file into memory.
#[derive(Debug, Clone, Copy)]
pub struct FileReadRequest<'a> {
    /// Path of the file to read.
    pub path: &'a Path,
}

impl<'a> FileReadRequest<'a> {
    /// Create a new file read request.
    #[inline]
    #[must_use]
    pub fn new(path: &'a Path) -> Self {
        Self { path }
    }
}

// ============================================================================
// RangeReadRequest
// ============================================================================

/// Request to read a contiguous byte range from a file.
#[derive(Debug, Clone, Copy)]
pub struct RangeReadRequest<'a> {
    /// Path of the file to read from.
    pub path: &'a Path,
    /// Byte offset at which to start reading.
    pub offset: u64,
    /// Number of bytes to read.
    pub len: usize,
}

impl<'a> RangeReadRequest<'a> {
    /// Create a new range read request.
    #[inline]
    #[must_use]
    pub fn new(path: &'a Path, offset: u64, len: usize) -> Self {
        Self { path, offset, len }
    }
}

// ============================================================================
// BatchReadRequest
// ============================================================================

/// A single range entry within a [`BatchReadRequest`].
#[derive(Debug, Clone)]
pub struct BatchRange {
    /// Byte offset within the file.
    pub offset: u64,
    /// Number of bytes to read.
    pub len: usize,
}

impl BatchRange {
    /// Create a new batch range entry.
    #[inline]
    #[must_use]
    pub fn new(offset: u64, len: usize) -> Self {
        Self { offset, len }
    }
}

/// Request to read multiple byte ranges, each tagged with a file path.
///
/// All ranges may come from the same file or from different files. Engines
/// that support coalescing may merge adjacent ranges automatically.
#[derive(Debug, Clone)]
pub struct BatchReadRequest<'a> {
    /// Parallel slices: `paths[i]` is the file for `ranges[i]`.
    pub paths: &'a [&'a Path],
    /// Ranges to read; `ranges[i]` corresponds to `paths[i]`.
    pub ranges: &'a [BatchRange],
}

impl<'a> BatchReadRequest<'a> {
    /// Create a new batch read request.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `paths.len() != ranges.len()`.
    #[inline]
    #[must_use]
    pub fn new(paths: &'a [&'a Path], ranges: &'a [BatchRange]) -> Self {
        debug_assert_eq!(
            paths.len(),
            ranges.len(),
            "BatchReadRequest: paths and ranges must have the same length"
        );
        Self { paths, ranges }
    }

    /// Number of ranges in this batch.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Returns `true` if the batch contains no ranges.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

// ============================================================================
// RangeReadResult
// ============================================================================

/// A single result from a [`BatchReadRequest`].
#[derive(Debug)]
pub struct RangeReadResult {
    /// Index into the original [`BatchReadRequest`] this result corresponds to.
    pub request_index: usize,
    /// The bytes that were read.
    pub bytes: std::sync::Arc<[u8]>,
    /// Byte offset within `bytes` where the requested data begins.
    ///
    /// Usually `0` for engines that return exactly the requested slice, but
    /// may be non-zero for engines that return a coalesced super-range.
    pub logical_offset: usize,
    /// Number of bytes of requested data starting at `logical_offset`.
    pub logical_len: usize,
}

impl RangeReadResult {
    /// Returns the requested data as a byte slice.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.bytes[self.logical_offset..self.logical_offset + self.logical_len]
    }
}

// ============================================================================
// WriteAtRequest
// ============================================================================

/// Request to write bytes at a specific offset in a file.
///
/// The file must already be open for writing by the engine. The engine is
/// responsible for managing file handles; this request carries only the data
/// and destination position.
#[derive(Debug, Clone, Copy)]
pub struct WriteAtRequest<'a> {
    /// Byte offset within the file at which to begin writing.
    pub offset: u64,
    /// Data to write.
    pub data: &'a [u8],
}

impl<'a> WriteAtRequest<'a> {
    /// Create a new write-at request.
    #[inline]
    #[must_use]
    pub fn new(offset: u64, data: &'a [u8]) -> Self {
        Self { offset, data }
    }
}

// ============================================================================
// MmapRequest / MmapRangeRequest
// ============================================================================

/// Request to memory-map an entire file.
#[derive(Debug, Clone, Copy)]
pub struct MmapRequest<'a> {
    /// Path of the file to map.
    pub path: &'a Path,
}

impl<'a> MmapRequest<'a> {
    /// Create a new mmap request.
    #[inline]
    #[must_use]
    pub fn new(path: &'a Path) -> Self {
        Self { path }
    }
}

/// Request to memory-map a byte range within a file.
#[derive(Debug, Clone, Copy)]
pub struct MmapRangeRequest<'a> {
    /// Path of the file to map.
    pub path: &'a Path,
    /// Byte offset at which the mapping begins.
    pub offset: u64,
    /// Length of the mapped region in bytes.
    pub len: usize,
}

impl<'a> MmapRangeRequest<'a> {
    /// Create a new mmap range request.
    #[inline]
    #[must_use]
    pub fn new(path: &'a Path, offset: u64, len: usize) -> Self {
        Self { path, offset, len }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn file_read_request_stores_path() {
        let p = Path::new("/tmp/model.safetensors");
        let req = FileReadRequest::new(p);
        assert_eq!(req.path, p);
    }

    #[test]
    fn range_read_request_fields() {
        let p = Path::new("/tmp/shard.bin");
        let req = RangeReadRequest::new(p, 1024, 512);
        assert_eq!(req.path, p);
        assert_eq!(req.offset, 1024);
        assert_eq!(req.len, 512);
    }

    #[test]
    fn batch_read_request_len_and_is_empty() {
        let p = Path::new("/tmp/x");
        let paths = [p];
        let ranges = [BatchRange::new(0, 64)];
        let req = BatchReadRequest::new(&paths, &ranges);
        assert_eq!(req.len(), 1);
        assert!(!req.is_empty());

        let empty = BatchReadRequest::new(&[], &[]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn range_read_result_data_slice() {
        let bytes: std::sync::Arc<[u8]> = std::sync::Arc::from(vec![0u8, 1, 2, 3, 4, 5]);
        let result = RangeReadResult {
            request_index: 0,
            bytes,
            logical_offset: 2,
            logical_len: 3,
        };
        assert_eq!(result.data(), &[2, 3, 4]);
    }

    #[test]
    fn write_at_request_fields() {
        let data = b"hello";
        let req = WriteAtRequest::new(4096, data);
        assert_eq!(req.offset, 4096);
        assert_eq!(req.data, b"hello");
    }

    #[test]
    fn mmap_request_stores_path() {
        let p = Path::new("/tmp/weights.bin");
        let req = MmapRequest::new(p);
        assert_eq!(req.path, p);
    }

    #[test]
    fn mmap_range_request_fields() {
        let p = Path::new("/tmp/partition_0");
        let req = MmapRangeRequest::new(p, 512, 4096);
        assert_eq!(req.path, p);
        assert_eq!(req.offset, 512);
        assert_eq!(req.len, 4096);
    }

    #[test]
    fn batch_range_new() {
        let r = BatchRange::new(8192, 1024);
        assert_eq!(r.offset, 8192);
        assert_eq!(r.len, 1024);
    }
}
