//! O_DIRECT file operations and aligned buffer management for Linux.
//!
//! Provides utilities for direct I/O that bypasses the kernel page cache.

use super::IoResult;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::OpenOptions as StdOpenOptions;
use std::io::{Error as IoError, ErrorKind};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::ptr::NonNull;

pub const BLOCK_SIZE: usize = 4096;
pub const BLOCK_SIZE_U64: u64 = 4096;

#[inline]
pub const fn is_block_aligned(offset: u64, len: usize) -> bool {
    (offset & (BLOCK_SIZE_U64 - 1)) == 0 && len.is_multiple_of(BLOCK_SIZE)
}

#[inline]
pub const fn can_use_direct_read(file_size: usize, chunk_size: usize) -> bool {
    file_size > 0
        && is_block_aligned(0, file_size)
        && chunk_size.is_multiple_of(BLOCK_SIZE)
        && file_size.is_multiple_of(chunk_size)
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
