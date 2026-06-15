//! Memory-mapped storage engine.
//!
//! [`MmapStorage`] implements [`MappableStorage`] and [`StorageEngine`].
//! It uses `memmap2` to create page-aligned, read-only memory mappings, letting
//! the OS manage paging rather than copying file data into user-space buffers.
//!
//! Obtain a mapping with [`MmapStorage::new().map_file`] or
//! [`MmapStorage::new().map_range`]; the returned [`MmapRegion`] is
//! cheaply cloneable (backed by an `Arc`) and implements `AsRef<[u8]>`.

use crate::storage::{
    ByteRange, IoResult,
    availability::{StorageAvailability, StorageKind},
    buffer::MmapRegion,
};
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// MmapStorage
// ============================================================================

/// Storage engine for memory-mapped file access.
///
/// This engine is a pure zero-copy reader: it maps files (or ranges within
/// files) into the process address space and lets the OS page data in on
/// demand.  It does **not** implement `ReadableStorage` or `WritableStorage`.
///
/// ```rust,ignore
/// use tensora::storage::mmap::MmapStorage;
/// use tensora::storage::{MappableStorage};
/// use std::path::Path;
///
/// let engine = MmapStorage::new();
/// let region = engine.map_file(Path::new("model.bin"))?;
/// let data: &[u8] = region.as_slice();
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MmapStorage;

impl MmapStorage {
    /// Create a new `MmapStorage` engine.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

// ============================================================================
// StorageEngine impl
// ============================================================================

impl super::StorageEngine for MmapStorage {
    const KIND: StorageKind = StorageKind::Mmap;

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        // mmap is available whenever the OS returns a non-zero page size.
        use crate::storage::availability::UnavailableReason;
        let page_size = region::page::size();
        if page_size == 0 {
            return StorageAvailability::unavailable(
                UnavailableReason::MissingDependency,
                "region returned a zero page size",
            );
        }
        StorageAvailability::Available
    }
}

// ============================================================================
// MappableStorage impl
// ============================================================================

impl super::MappableStorage for MmapStorage {
    /// Memory-map an entire file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened, is empty, or exceeds
    /// `usize::MAX` bytes.
    fn map_file(&self, path: &Path) -> IoResult<MmapRegion> {
        let file = File::open(path)?;
        let len_u64 = file.metadata()?.len();
        let len = usize::try_from(len_u64)
            .map_err(|_| Error::new(ErrorKind::InvalidInput, "file too large"))?;
        if len == 0 {
            return Err(Error::new(ErrorKind::InvalidData, "cannot mmap empty file"));
        }
        // SAFETY: the file is opened read-only and remains alive for the mapping
        // call; memmap2 owns the mapping after creation.
        let inner = unsafe { MmapOptions::new().map(&file)? };
        Ok(MmapRegion::new(Arc::new(inner), 0, len))
    }

    /// Memory-map a byte range within a file.
    ///
    /// The mapping is page-aligned internally; `region.as_slice()` returns
    /// exactly `range.len()` bytes starting at `range.start()`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened, `range` is empty,
    /// the range overflows, or the range extends beyond the file.
    fn map_range(&self, path: &Path, range: ByteRange) -> IoResult<MmapRegion> {
        if range.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "cannot mmap empty range",
            ));
        }
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let end = range.end();
        if end > file_len {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "range exceeds file size",
            ));
        }
        let page_size = u64::try_from(region::page::size()).map_err(|e| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("invalid system page size: {e}"),
            )
        })?;
        if page_size == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "system page size must be non-zero",
            ));
        }
        let aligned_offset = (range.start() / page_size) * page_size;
        let start = usize::try_from(range.start() - aligned_offset)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, e))?;
        let len = range.len_usize()?;
        let map_len = len
            .checked_add(start)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "mapping length overflow"))?;
        // SAFETY: the requested range has been bounds-checked against file_len,
        // and the offset passed to memmap2 is page-aligned as required.
        let inner = unsafe {
            MmapOptions::new()
                .offset(aligned_offset)
                .len(map_len)
                .map(&file)?
        };
        Ok(MmapRegion::new(Arc::new(inner), start, len))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{ByteRange, MappableStorage, StorageEngine};
    use tempfile::TempDir;

    fn write_tmp(dir: &TempDir, name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn kind_is_mmap() {
        assert_eq!(MmapStorage::new().kind(), StorageKind::Mmap);
    }

    #[test]
    fn availability_is_available() {
        assert!(MmapStorage::availability().is_available());
    }

    #[test]
    fn map_file_roundtrip() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let path = write_tmp(&dir, "file.bin", &data);

        let region = MmapStorage::new().map_file(&path).unwrap();
        assert_eq!(region.as_slice(), &data[..]);
        assert_eq!(region.len(), 4096);
        assert!(!region.is_empty());
    }

    #[test]
    fn map_file_empty_returns_error() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "empty.bin", b"");

        let err = MmapStorage::new().map_file(&path).unwrap_err();
        assert!(
            err.kind() == std::io::ErrorKind::InvalidData
                || err.kind() == std::io::ErrorKind::Other,
            "unexpected error kind: {:?}",
            err.kind()
        );
    }

    #[test]
    fn map_range_returns_correct_slice() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..100).collect();
        let path = write_tmp(&dir, "range.bin", &data);

        let region = MmapStorage::new()
            .map_range(&path, ByteRange::from_offset_len(10, 20).unwrap())
            .unwrap();
        assert_eq!(region.as_slice(), &data[10..30]);
        assert_eq!(region.len(), 20);
    }

    #[test]
    fn map_range_zero_len_returns_error() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "z.bin", b"hello");

        let err = MmapStorage::new()
            .map_range(&path, ByteRange::from_offset_len(0, 0).unwrap())
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn map_range_beyond_file_returns_error() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "short.bin", b"hi");

        let err = MmapStorage::new()
            .map_range(&path, ByteRange::from_offset_len(0, 1000).unwrap())
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn map_file_large() {
        let dir = TempDir::new().unwrap();
        let data = vec![0xABu8; 1024 * 1024]; // 1 MiB
        let path = write_tmp(&dir, "large.bin", &data);

        let region = MmapStorage::new().map_file(&path).unwrap();
        assert_eq!(region.len(), data.len());
        assert!(region.as_slice().iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn map_range_page_boundary() {
        let dir = TempDir::new().unwrap();
        // Write 2 pages worth of data (assuming ≤ 4 KiB pages)
        let mut data = vec![0u8; 8192];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let path = write_tmp(&dir, "pages.bin", &data);

        // Map 100 bytes starting at the second page boundary (4096)
        let region = MmapStorage::new()
            .map_range(&path, ByteRange::from_offset_len(4096, 100).unwrap())
            .unwrap();
        assert_eq!(region.len(), 100);
        assert_eq!(region.as_slice(), &data[4096..4196]);
    }

    #[test]
    fn region_is_cheaply_cloneable() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "clone.bin", b"hello clone");

        let r1 = MmapStorage::new().map_file(&path).unwrap();
        let r2 = r1.clone();
        assert_eq!(r1.as_slice(), r2.as_slice());
        assert_eq!(r2.as_slice(), b"hello clone");
    }
}
