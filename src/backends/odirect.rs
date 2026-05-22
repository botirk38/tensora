//! O_DIRECT file operations and aligned buffer management for Linux.
//!
//! Provides utilities for direct I/O that bypasses the kernel page cache.

use super::IoResult;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fs::OpenOptions as StdOpenOptions;
use std::io::{Error as IoError, ErrorKind};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::ptr::NonNull;

pub const BLOCK_SIZE: usize = 4096;
pub const BLOCK_SIZE_U64: u64 = 4096;

#[cfg(test)]
#[inline]
pub const fn is_block_aligned(offset: u64, len: usize) -> bool {
    (offset & (BLOCK_SIZE_U64 - 1)) == 0 && len.is_multiple_of(BLOCK_SIZE)
}

#[inline]
pub const fn round_up_to_block(n: usize) -> usize {
    (n + BLOCK_SIZE - 1) & !(BLOCK_SIZE - 1)
}

#[allow(dead_code)]
#[inline]
pub const fn round_up_to_block_u64(n: u64) -> u64 {
    (n + BLOCK_SIZE_U64 - 1) & !(BLOCK_SIZE_U64 - 1)
}

pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
}

impl AlignedBuffer {
    pub fn new(capacity: usize) -> IoResult<Self> {
        if capacity == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                layout: unsafe { Layout::from_size_align_unchecked(1, 1) },
                len: 0,
            });
        }

        let layout = Layout::from_size_align(capacity, BLOCK_SIZE)
            .map_err(|_| IoError::new(ErrorKind::InvalidInput, "invalid allocation layout"))?;

        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            layout,
            len: 0,
        })
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.layout.size());
        self.len = len;
    }

    pub fn capacity(&self) -> usize {
        self.layout.size()
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
        }
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len)
            .field("capacity", &self.capacity())
            .finish()
    }
}

unsafe impl Send for AlignedBuffer {}

#[inline]
pub fn alloc_aligned(capacity: usize) -> IoResult<AlignedBuffer> {
    AlignedBuffer::new(capacity)
}

#[inline]
pub fn open_direct_read(path: &Path) -> IoResult<std::fs::File> {
    StdOpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

/// Try O_DIRECT, fall back to regular open if the filesystem doesn't support it.
#[inline]
pub fn open_prefer_direct(path: &Path) -> IoResult<(std::fs::File, bool)> {
    match open_direct_read(path) {
        Ok(f) => Ok((f, true)),
        Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
            let f = std::fs::File::open(path)?;
            Ok((f, false))
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_block_aligned_true() {
        assert!(is_block_aligned(0, BLOCK_SIZE));
        assert!(is_block_aligned(BLOCK_SIZE_U64, BLOCK_SIZE * 2));
    }

    #[test]
    fn is_block_aligned_false_offset() {
        assert!(!is_block_aligned(1, BLOCK_SIZE));
        assert!(!is_block_aligned(BLOCK_SIZE_U64 - 1, BLOCK_SIZE));
    }

    #[test]
    fn is_block_aligned_false_len() {
        assert!(!is_block_aligned(0, BLOCK_SIZE - 1));
        assert!(!is_block_aligned(0, 1));
    }

    #[test]
    fn round_up_to_block_aligned() {
        assert_eq!(round_up_to_block(BLOCK_SIZE), BLOCK_SIZE);
        assert_eq!(round_up_to_block(BLOCK_SIZE * 4), BLOCK_SIZE * 4);
    }

    #[test]
    fn round_up_to_block_unaligned() {
        assert_eq!(round_up_to_block(1), BLOCK_SIZE);
        assert_eq!(round_up_to_block(BLOCK_SIZE + 1), BLOCK_SIZE * 2);
    }

    #[test]
    fn round_up_to_block_zero() {
        assert_eq!(round_up_to_block(0), 0);
    }

    #[test]
    fn aligned_buffer_set_len_and_slice() {
        let mut buf = AlignedBuffer::new(BLOCK_SIZE).unwrap();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), BLOCK_SIZE);

        buf.set_len(16);
        assert_eq!(buf.len(), 16);

        let slice = buf.as_mut_slice();
        slice[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    #[test]
    #[should_panic]
    fn aligned_buffer_set_len_exceeds_capacity() {
        let mut buf = AlignedBuffer::new(BLOCK_SIZE).unwrap();
        buf.set_len(BLOCK_SIZE + 1);
    }

    #[test]
    fn aligned_buffer_debug() {
        let buf = AlignedBuffer::new(BLOCK_SIZE).unwrap();
        let dbg = format!("{:?}", buf);
        assert!(dbg.contains("AlignedBuffer"));
        assert!(dbg.contains("len"));
    }

    #[test]
    fn alloc_aligned_works() {
        let buf = alloc_aligned(BLOCK_SIZE).unwrap();
        assert_eq!(buf.capacity(), BLOCK_SIZE);
        assert_eq!(buf.len(), 0);
    }
}
