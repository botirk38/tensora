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
