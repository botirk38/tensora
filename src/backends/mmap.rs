  //! Memory-mapped I/O backend for zero-copy file access.
//!
//! This module provides memory-mapped file loading using `memmap2`, allowing
//! the OS to manage file data paging. Useful for lazy loading and random access patterns.
//!
//! # Features
//!
//! - Zero-copy access to file data
//! - OS-managed memory paging
//! - Suitable for read-only access
//! - Good for random access patterns
//!
//! # Example
//!
//! ```ignore
//! use tensor_store::backends::mmap;
//!
//! let mmap = mmap::load("large_file.bin")?;
//! let data = mmap.as_slice(); // No copy, just pointer
//! ```

use super::IoResult;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;
use std::sync::Arc;

use memmap2::MmapOptions;
use region::page;

/// Memory-mapped file region (zero-copy, lazy)
#[derive(Debug, Clone)]
pub struct Mmap {
    pub inner: Arc<memmap2::Mmap>,
    pub start: usize, // offset within mmap where data starts
    pub len: usize,   // length of actual data
}

impl Mmap {
    /// Get slice view of the mapped data
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.inner[self.start..self.start + self.len]
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl AsRef<[u8]> for Mmap {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::Deref for Mmap {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// Map entire file into memory
pub fn map(path: impl AsRef<Path>) -> IoResult<Mmap> {
    let file = File::open(path.as_ref())?;
    let len = file.metadata()?.len();

    let len_usize = usize::try_from(len)
        .map_err(|_foo| Error::new(ErrorKind::InvalidInput, "file too large"))?;

    if len_usize == 0 {
        return Err(Error::new(ErrorKind::InvalidData, "cannot mmap empty file"));
    }

    let inner = unsafe { MmapOptions::new().map(&file)? };

    Ok(Mmap {
        inner: Arc::new(inner),
        start: 0,
        len: len_usize,
    })
}

/// Map a byte range from file into memory
pub fn map_range(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Mmap> {
    if len == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "cannot mmap empty range",
        ));
    }

    let file = File::open(path.as_ref())?;
    let file_len = file.metadata()?.len();

    // Validate range
    let end = offset
        .checked_add(
            u64::try_from(len)
                .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("len too large: {e}")))?,
        )
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "range overflow"))?;
    if end > file_len {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "range exceeds file size",
        ));
    }

    // Calculate page-aligned mapping
    let page_size = u64::try_from(page::size()).unwrap_or(4096);
    let aligned_offset = (offset / page_size) * page_size;
    let start = usize::try_from(offset - aligned_offset)
        .map_err(|e| Error::new(ErrorKind::InvalidInput, e))?;
    let map_len = len + start;

    let inner = unsafe {
        MmapOptions::new()
            .offset(aligned_offset)
            .len(map_len)
            .map(&file)?
    };

    Ok(Mmap {
        inner: Arc::new(inner),
        start,
        len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_basic() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"Hello, mmap!").unwrap();
        tmpfile.flush().unwrap();

        let mmap = map(tmpfile.path()).unwrap();
        assert_eq!(mmap.as_slice(), b"Hello, mmap!");
        assert_eq!(mmap.len(), 12);
        assert!(!mmap.is_empty());
    }

    #[test]
    fn test_mmap_empty_file_error() {
        let tmpfile = NamedTempFile::new().unwrap();
        // File is empty
        let result = map(tmpfile.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_large_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = vec![42u8; 10 * 1024 * 1024]; // 10MB
        tmpfile.write_all(&data).unwrap();
        tmpfile.flush().unwrap();

        let mmap = map(tmpfile.path()).unwrap();
        assert_eq!(mmap.len(), 10 * 1024 * 1024);
        assert!(mmap.as_slice().iter().all(|&x| x == 42));
    }

    #[test]
    fn test_mmap_range_aligned() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"0123456789ABCDEF").unwrap();
        tmpfile.flush().unwrap();

        let mmap = map_range(tmpfile.path(), 4, 8).unwrap();
        assert_eq!(mmap.as_slice(), b"456789AB");
        assert_eq!(mmap.len(), 8);
    }

    #[test]
    fn test_mmap_range_unaligned() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"0123456789ABCDEF").unwrap();
        tmpfile.flush().unwrap();

        // Unaligned offset
        let mmap = map_range(tmpfile.path(), 3, 5).unwrap();
        assert_eq!(mmap.as_slice(), b"34567");
        assert_eq!(mmap.len(), 5);
    }

    #[test]
    fn test_mmap_range_page_boundary() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let mut data = vec![0u8; 8192]; // 2 pages (assuming 4KB pages)
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        tmpfile.write_all(&data).unwrap();
        tmpfile.flush().unwrap();

        // Map from page boundary
        let mmap = map_range(tmpfile.path(), 4096, 100).unwrap();
        assert_eq!(mmap.len(), 100);
        assert_eq!(mmap.as_slice()[0], 0); // First byte of second page
    }

    #[test]
    fn test_mmap_range_exceeds_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"short").unwrap();
        tmpfile.flush().unwrap();

        let result = map_range(tmpfile.path(), 0, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_range_zero_length_error() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"data").unwrap();
        tmpfile.flush().unwrap();

        let result = map_range(tmpfile.path(), 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_clone() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"clone test").unwrap();
        tmpfile.flush().unwrap();

        let mmap1 = map(tmpfile.path()).unwrap();
        let mmap2 = mmap1.clone();

        assert_eq!(mmap1.as_slice(), mmap2.as_slice());
        assert_eq!(mmap1.len(), mmap2.len());
        assert_eq!(Arc::strong_count(&mmap1.inner), 2);
    }

    #[test]
    fn test_mmap_deref() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"deref test").unwrap();
        tmpfile.flush().unwrap();

        let mmap = map(tmpfile.path()).unwrap();

        // Test Deref trait
        let slice: &[u8] = &*mmap;
        assert_eq!(slice, b"deref test");
    }

    #[test]
    fn test_mmap_as_ref() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"as_ref test").unwrap();
        tmpfile.flush().unwrap();

        let mmap = map(tmpfile.path()).unwrap();

        // Test AsRef trait
        let slice: &[u8] = mmap.as_ref();
        assert_eq!(slice, b"as_ref test");
    }

    #[test]
    fn test_mmap_range_full_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = b"full file range";
        tmpfile.write_all(data).unwrap();
        tmpfile.flush().unwrap();

        let mmap = map_range(tmpfile.path(), 0, data.len()).unwrap();
        assert_eq!(mmap.as_slice(), data);
    }
}
