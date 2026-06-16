//! Storage engine traits and vocabulary types.
//!
//! # Traits
//!
//! - [`StorageEngine`] — base trait all engines implement (kind, availability)
//! - [`ReadableStorage`] — exact file and range reads
//! - [`WritableStorage`] — path-based writes and durability controls
//! - [`AsyncReadableStorage`] — async exact file and range reads
//! - [`AsyncWritableStorage`] — async path-based writes and durability controls
//! - [`MappableStorage`] — memory-map a file or range (mmap engine only)
//!
//! # Vocabulary types
//!
//! - [`ByteRange`] — validated half-open byte range `[start, end)`
//! - [`FileRange`] — path plus byte range for batch reads
//! - [`RangeRead`] — single result from a batch read
//! - [`WriteSlice`] — positioned write entry
//!
//! # Writing
//!
//! Callers pass a path; the engine owns file-handle lifetime. Methods divide
//! into two groups:
//!
//! - **Creating files**: [`WritableStorage::write_file`] and
//!   [`WritableStorage::write_positioned_file`] create or truncate the path.
//! - **Updating files**: [`WritableStorage::write_at`] and
//!   [`WritableStorage::write_slices`] open an existing file without truncating.
//!
//! Durability is explicit: call [`WritableStorage::sync_data`] or
//! [`WritableStorage::sync_all`] after writes that must survive a crash.

pub mod availability;
pub mod buffer;
#[cfg(target_os = "linux")]
pub mod io_uring;
pub mod mmap;
pub mod sync;
pub mod tokio;

pub use std::io::Result as IoResult;

use std::io::{Error, ErrorKind};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    start: u64,
    end: u64,
}

impl ByteRange {
    #[inline]
    pub fn new(start: u64, end: u64) -> IoResult<Self> {
        if end < start {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "byte range end is before start",
            ));
        }
        Ok(Self { start, end })
    }

    #[inline]
    pub fn from_offset_len(offset: u64, len: usize) -> IoResult<Self> {
        let len = u64::try_from(len).map_err(|e| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("range length too large: {e}"),
            )
        })?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "byte range overflow"))?;
        Self::new(offset, end)
    }

    #[inline]
    #[must_use]
    pub const fn start(self) -> u64 {
        self.start
    }

    #[inline]
    #[must_use]
    pub const fn end(self) -> u64 {
        self.end
    }

    #[inline]
    #[must_use]
    pub const fn len(self) -> u64 {
        self.end - self.start
    }

    #[inline]
    pub fn len_usize(self) -> IoResult<usize> {
        usize::try_from(self.len()).map_err(|e| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("range length too large: {e}"),
            )
        })
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FileRange<'a> {
    pub path: &'a Path,
    pub range: ByteRange,
}

impl<'a> FileRange<'a> {
    #[inline]
    #[must_use]
    pub const fn new(path: &'a Path, range: ByteRange) -> Self {
        Self { path, range }
    }
}

#[derive(Debug)]
pub struct RangeRead {
    pub request_index: usize,
    pub range: ByteRange,
    pub bytes: Arc<[u8]>,
}

impl RangeRead {
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WriteSlice<'a> {
    pub offset: u64,
    pub data: &'a [u8],
}

impl<'a> WriteSlice<'a> {
    #[inline]
    #[must_use]
    pub const fn new(offset: u64, data: &'a [u8]) -> Self {
        Self { offset, data }
    }

    #[inline]
    pub fn end_offset(self) -> IoResult<u64> {
        let len = u64::try_from(self.data.len()).map_err(|e| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("write length too large: {e}"),
            )
        })?;
        self.offset
            .checked_add(len)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "write offset overflow"))
    }
}

/// A validated, non-overlapping slice of [`WriteSlice`] entries.
///
/// Construct with [`WriteSlices::new`], which validates that no two entries
/// overlap and that no entry overflows `u64`.  Backends receive `WriteSlices`
/// and can assume the invariant holds without re-checking.
#[derive(Debug, Clone, Copy)]
pub struct WriteSlices<'a>(pub &'a [WriteSlice<'a>]);

impl<'a> WriteSlices<'a> {
    /// Validates `slices` and wraps them.
    ///
    /// Returns `InvalidInput` if any slice overflows `u64` or if any two
    /// slices have overlapping byte ranges.  An empty slice is always valid.
    pub fn new(slices: &'a [WriteSlice<'a>]) -> IoResult<Self> {
        let mut sorted: Vec<(u64, u64)> = slices
            .iter()
            .map(|w| w.end_offset().map(|end| (w.offset, end)))
            .collect::<IoResult<_>>()?;
        sorted.sort_unstable_by_key(|&(s, _)| s);
        for pair in sorted.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(Error::new(ErrorKind::InvalidInput, "write slices overlap"));
            }
        }
        Ok(Self(slices))
    }

    /// Wraps `slices` without validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the slices are non-overlapping and that
    /// no slice overflows `u64`.  Violating this is not UB but will cause
    /// incorrect (non-deterministic) writes when backends parallelize.
    #[inline]
    #[must_use]
    pub unsafe fn new_unchecked(slices: &'a [WriteSlice<'a>]) -> Self {
        Self(slices)
    }

    /// Returns the inner slice.  Guaranteed non-overlapping.
    #[inline]
    #[must_use]
    pub fn as_slice(self) -> &'a [WriteSlice<'a>] {
        self.0
    }

    /// Returns `true` if there are no slices.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }
}

/// Common metadata every storage engine exposes.
pub trait StorageEngine {
    /// Compile-time kind identifier for this engine.
    const KIND: availability::StorageKind;

    /// Returns the kind identifier for this engine value.
    fn kind(&self) -> availability::StorageKind {
        Self::KIND
    }

    /// Reports whether this engine can run in the current environment.
    fn availability() -> availability::StorageAvailability
    where
        Self: Sized;
}

/// Blocking storage engine operations for exact reads.
pub trait ReadableStorage: StorageEngine {
    /// Reads an entire file into owned bytes.
    fn read_file(&self, path: &Path) -> IoResult<buffer::OwnedBytes>;

    /// Reads exactly `range` from `path`.
    ///
    /// Empty ranges are valid and return empty bytes.
    fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<buffer::OwnedBytes>;

    /// Reads a batch of file ranges concurrently.
    ///
    /// Each backend uses its own parallel strategy (Rayon, io_uring ring
    /// saturation, etc.).  Results preserve the original `request_index` of
    /// each entry.
    fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>>;
}

/// Async storage engine operations for exact reads.
#[allow(async_fn_in_trait)]
pub trait AsyncReadableStorage: StorageEngine {
    /// Reads an entire file into owned bytes.
    async fn read_file(&self, path: &Path) -> IoResult<buffer::OwnedBytes>;

    /// Reads exactly `range` from `path`.
    ///
    /// Empty ranges are valid and return empty bytes.
    async fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<buffer::OwnedBytes>;

    /// Reads a batch of file ranges.
    async fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>>;
}

/// Blocking path-based write operations.
///
/// The engine owns file-handle lifetime for every operation.  Callers never
/// open or close files; they only supply paths and byte slices.
///
/// ## Creating vs. updating
///
/// - [`write_file`](WritableStorage::write_file) and
///   [`write_positioned_file`](WritableStorage::write_positioned_file) always
///   create or truncate the target path.
/// - [`write_at`](WritableStorage::write_at) and
///   [`write_slices`](WritableStorage::write_slices) open an existing file and
///   write without truncating.  They do not create the file if it is absent.
pub trait WritableStorage: StorageEngine {
    /// Creates or truncates `path` and writes all of `data` at offset 0.
    ///
    /// Does not sync; call [`sync_data`](WritableStorage::sync_data) or
    /// [`sync_all`](WritableStorage::sync_all) when durability is required.
    fn write_file(&self, path: &Path, data: &[u8]) -> IoResult<()>;

    /// Creates or truncates `path`, sets its length to `len`, then applies
    /// every slice in `writes` in parallel.
    ///
    /// `writes` is pre-validated (non-overlapping, no overflow).  Does not sync.
    fn write_positioned_file(&self, path: &Path, len: u64, writes: WriteSlices<'_>)
    -> IoResult<()>;

    /// Opens an existing file at `path` and writes `data` starting at `offset`.
    ///
    /// Does not create or truncate the file.  Must write the full slice or
    /// return an error.
    fn write_at(&self, path: &Path, offset: u64, data: &[u8]) -> IoResult<()>;

    /// Opens an existing file at `path` and applies every slice in `writes`
    /// in parallel.
    ///
    /// `writes` is pre-validated (non-overlapping, no overflow).  Does not
    /// create or truncate the file.
    fn write_slices(&self, path: &Path, writes: WriteSlices<'_>) -> IoResult<()>;

    /// Syncs file data (but not metadata) to durable storage.
    fn sync_data(&self, path: &Path) -> IoResult<()>;

    /// Syncs file data and metadata to durable storage.
    fn sync_all(&self, path: &Path) -> IoResult<()>;
}

/// Async path-based write operations.
///
/// Mirrors [`WritableStorage`] with async methods.  The engine owns
/// file-handle lifetime; callers never open or close files.
#[allow(async_fn_in_trait)]
pub trait AsyncWritableStorage: StorageEngine {
    /// Creates or truncates `path` and writes all of `data` at offset 0.
    async fn write_file(&self, path: &Path, data: &[u8]) -> IoResult<()>;

    /// Creates or truncates `path`, sets its length to `len`, then applies
    /// every slice in `writes` concurrently.
    ///
    /// `writes` is pre-validated (non-overlapping, no overflow).  Does not sync.
    async fn write_positioned_file(
        &self,
        path: &Path,
        len: u64,
        writes: WriteSlices<'_>,
    ) -> IoResult<()>;

    /// Opens an existing file at `path` and writes `data` starting at `offset`.
    async fn write_at(&self, path: &Path, offset: u64, data: &[u8]) -> IoResult<()>;

    /// Opens an existing file at `path` and applies every slice in `writes`
    /// concurrently.
    ///
    /// `writes` is pre-validated (non-overlapping, no overflow).  Does not
    /// create or truncate the file.
    async fn write_slices(&self, path: &Path, writes: WriteSlices<'_>) -> IoResult<()>;

    /// Syncs file data (but not metadata) to durable storage.
    async fn sync_data(&self, path: &Path) -> IoResult<()>;

    /// Syncs file data and metadata to durable storage.
    async fn sync_all(&self, path: &Path) -> IoResult<()>;
}

/// Storage engine operations for memory mapping files.
pub trait MappableStorage: StorageEngine {
    /// Maps the entire file at `path`.
    fn map_file(&self, path: &Path) -> IoResult<buffer::MmapRegion>;

    /// Maps exactly `range` from `path`.
    ///
    /// Empty ranges are rejected because zero-length memory maps are not
    /// portable.
    fn map_range(&self, path: &Path, range: ByteRange) -> IoResult<buffer::MmapRegion>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn byte_range_new_validates_order() {
        let range = ByteRange::new(10, 20).unwrap();
        assert_eq!(range.start(), 10);
        assert_eq!(range.end(), 20);
        assert_eq!(range.len(), 10);
        assert!(!range.is_empty());
        assert!(ByteRange::new(20, 10).is_err());
    }

    #[test]
    fn byte_range_from_offset_len_validates_overflow() {
        let range = ByteRange::from_offset_len(10, 5).unwrap();
        assert_eq!(range, ByteRange::new(10, 15).unwrap());
        assert!(ByteRange::from_offset_len(u64::MAX, 1).is_err());
    }

    #[test]
    fn byte_range_allows_empty_reads() {
        let range = ByteRange::from_offset_len(42, 0).unwrap();
        assert!(range.is_empty());
        assert_eq!(range.len_usize().unwrap(), 0);
    }

    #[test]
    fn file_range_stores_path_and_range() {
        let path = Path::new("/tmp/shard.bin");
        let range = ByteRange::new(1, 4).unwrap();
        let file_range = FileRange::new(path, range);
        assert_eq!(file_range.path, path);
        assert_eq!(file_range.range, range);
    }

    #[test]
    fn range_read_data_slice() {
        let bytes: Arc<[u8]> = Arc::from(vec![2u8, 3, 4]);
        let range = ByteRange::new(0, 3).unwrap();
        let result = RangeRead {
            request_index: 0,
            range,
            bytes,
        };
        assert_eq!(result.data(), &[2, 3, 4]);
    }

    #[test]
    fn write_slice_end_offset_validates_overflow() {
        assert_eq!(WriteSlice::new(10, b"abc").end_offset().unwrap(), 13);
        assert!(WriteSlice::new(u64::MAX, b"x").end_offset().is_err());
    }

    #[test]
    fn write_slices_empty_is_valid() {
        assert!(WriteSlices::new(&[]).is_ok());
    }

    #[test]
    fn write_slices_non_overlapping_is_valid() {
        let a = WriteSlice::new(0, b"HELLO");
        let b = WriteSlice::new(10, b"WORLD");
        assert!(WriteSlices::new(&[a, b]).is_ok());
    }

    #[test]
    fn write_slices_adjacent_is_valid() {
        // end of first == start of second: not overlapping
        let a = WriteSlice::new(0, b"HELLO");
        let b = WriteSlice::new(5, b"WORLD");
        assert!(WriteSlices::new(&[a, b]).is_ok());
    }

    #[test]
    fn write_slices_overlapping_is_err() {
        let a = WriteSlice::new(0, b"AAAAA");
        let b = WriteSlice::new(3, b"BBBBB");
        let err = WriteSlices::new(&[a, b]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn write_slices_overflow_is_err() {
        let a = WriteSlice::new(u64::MAX, b"x");
        assert!(WriteSlices::new(&[a]).is_err());
    }

    #[test]
    fn write_slices_order_independent_validation() {
        // provided in reverse order — sort should still detect overlap
        let a = WriteSlice::new(5, b"BBBBB");
        let b = WriteSlice::new(2, b"AAAAA");
        // b spans [2,7), a spans [5,10) — overlap
        let err = WriteSlices::new(&[a, b]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }
}
