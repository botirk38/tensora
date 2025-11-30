//! O_DIRECT file operations and aligned buffer management for Linux.
//!
//! Provides utilities for direct I/O that bypasses the kernel page cache,
//! enabling true zero-copy operations when combined with io_uring.
//!
//! O_DIRECT requires buffers aligned to block boundaries (typically 4KB).
//! This module provides both the alignment utilities and the I/O operations.

use super::IoResult;
use std::alloc::{Layout, alloc_zeroed};
use std::fs::OpenOptions as StdOpenOptions;
use std::io::{Error as IoError, ErrorKind};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::Arc;
use tokio::fs::OpenOptions as TokioOpenOptions;
use tokio_uring::fs::{File as UringFile, OpenOptions};

/// Block size for alignment (4KB - standard disk block size)
pub const BLOCK_SIZE: usize = 4096;

/// Block size for direct I/O alignment (4KB) as u64.
pub const BLOCK_SIZE_U64: u64 = 4096;

// ---------------------------------------------------------------------------
// Alignment helpers
// ---------------------------------------------------------------------------

/// Rounds up a size to the next block-aligned boundary.
#[inline]
pub const fn align_to_block(size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let rem = size % BLOCK_SIZE;
    if rem == 0 {
        size
    } else {
        size + (BLOCK_SIZE - rem)
    }
}

/// Check if offset and length are properly aligned for O_DIRECT.
#[inline]
pub const fn is_block_aligned(offset: u64, len: usize) -> bool {
    (offset & (BLOCK_SIZE_U64 - 1)) == 0 && len.is_multiple_of(BLOCK_SIZE)
}

/// Determines whether a direct I/O read is safe for the given layout.
///
/// This requires:
/// - Non-zero file size
/// - File offset (0) and length block aligned
/// - Chunk size divides the file so each thread/task reads a whole number of chunks
#[inline]
pub const fn can_use_direct_read(file_size: usize, chunk_size: usize) -> bool {
    file_size > 0
        && is_block_aligned(0, file_size)
        && chunk_size.is_multiple_of(BLOCK_SIZE)
        && file_size.is_multiple_of(chunk_size)
}

/// Determines whether a direct I/O write is safe for the given length.
///
/// Direct writes require block-sized buffers; caller is responsible for alignment.
#[inline]
pub const fn can_use_direct_write(len: usize) -> bool {
    len.is_multiple_of(BLOCK_SIZE)
}

/// Align length for direct I/O write. Returns padded length.
#[inline]
pub const fn pad_to_block(len: usize) -> usize {
    align_to_block(len)
}

/// Allocates a zeroed, block-aligned buffer.
///
/// # Arguments
///
/// * `capacity` - Size in bytes (will be allocated with BLOCK_SIZE alignment)
///
/// # Returns
///
/// An empty Vec<u8> with the requested capacity, aligned to BLOCK_SIZE.
///
/// # Errors
///
/// Returns error if layout is invalid or allocation fails.
#[inline]
pub fn alloc_aligned(capacity: usize) -> IoResult<Vec<u8>> {
    if capacity == 0 {
        return Ok(Vec::new());
    }

    let layout = Layout::from_size_align(capacity, BLOCK_SIZE)
        .map_err(|_| IoError::new(ErrorKind::InvalidInput, "invalid allocation layout"))?;

    // SAFETY: We allocate zeroed memory to avoid exposing uninitialized data
    // when we truncate padded buffers to actual data size.
    let ptr = unsafe { alloc_zeroed(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }

    // SAFETY: ptr is non-null and allocated with the correct layout.
    // We start with length 0 to allow safe initialization.
    let buf = unsafe { Vec::from_raw_parts(ptr, 0, capacity) };
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Core aligned buffer types
// ---------------------------------------------------------------------------

/// The actual backing storage for aligned buffers.
///
/// Owns a block-aligned Vec<u8> and provides safe access to it.
struct AlignedBacking {
    buf: Vec<u8>,
}

impl AlignedBacking {
    /// Creates a new aligned backing buffer with the specified capacity.
    fn new(capacity: usize) -> IoResult<Self> {
        Ok(Self {
            buf: alloc_aligned(capacity)?,
        })
    }

    /// Returns the capacity of the backing buffer.
    #[inline]
    const fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Returns a mutable pointer to the start of the buffer.
    ///
    /// # Safety
    ///
    /// Caller must ensure proper bounds when using this pointer.
    #[inline]
    const fn base_ptr(&self) -> *mut u8 {
        self.buf.as_ptr().cast_mut()
    }

    /// Consumes the backing and returns the inner Vec with the specified length.
    ///
    /// # Errors
    ///
    /// Returns error if requested length exceeds capacity.
    fn into_vec(mut self, len: usize) -> IoResult<Vec<u8>> {
        if len > self.capacity() {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "requested length exceeds buffer capacity",
            ));
        }

        // SAFETY: len is bounded by capacity, and the allocation is valid.
        unsafe {
            self.buf.set_len(len);
        }
        Ok(self.buf)
    }
}

/// A view into a slice of an aligned buffer for concurrent I/O operations.
///
/// Multiple chunks can reference the same backing buffer at different offsets,
/// enabling true zero-copy parallel reads where each task writes directly to
/// its designated slice of the final buffer.
#[derive(Clone)]
pub struct AlignedChunk {
    backing: Arc<AlignedBacking>,
    offset: usize,
    len: usize,
    /// Number of initialized bytes (used by tokio-uring's IoBuf trait)
    init: usize,
}

impl AlignedChunk {
    /// Creates a new chunk view into the backing buffer.
    ///
    /// # Arguments
    ///
    /// * `backing` - Shared reference to the backing buffer
    /// * `offset` - Starting offset in bytes from the beginning of the backing buffer
    /// * `len` - Length of this chunk in bytes
    ///
    /// # Errors
    ///
    /// Returns error if the chunk would exceed the backing buffer's bounds.
    fn new(backing: Arc<AlignedBacking>, offset: usize, len: usize) -> IoResult<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| IoError::other("aligned chunk offset overflow"))?;

        if end > backing.capacity() {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "aligned chunk out of bounds",
            ));
        }

        Ok(Self {
            backing,
            offset,
            len,
            init: 0,
        })
    }

    /// Returns a mutable pointer to the start of this chunk.
    #[inline]
    fn ptr(&self) -> *mut u8 {
        // SAFETY: offset is validated during construction to be within bounds.
        unsafe { self.backing.base_ptr().add(self.offset) }
    }
}

// Implement tokio-uring's buffer traits for zero-copy I/O
unsafe impl tokio_uring::buf::IoBuf for AlignedChunk {
    fn stable_ptr(&self) -> *const u8 {
        self.ptr()
    }

    fn bytes_init(&self) -> usize {
        self.init
    }

    fn bytes_total(&self) -> usize {
        self.len
    }
}

unsafe impl tokio_uring::buf::IoBufMut for AlignedChunk {
    fn stable_mut_ptr(&mut self) -> *mut u8 {
        self.ptr()
    }

    unsafe fn set_init(&mut self, pos: usize) {
        assert!(
            pos <= self.len,
            "initialized length cannot exceed chunk length"
        );
        if pos > self.init {
            self.init = pos;
        }
    }
}

/// High-level aligned buffer that can be split into chunks for parallel I/O.
///
/// This is the main API for creating aligned buffers. It can be:
/// 1. Used directly for single I/O operations
/// 2. Split into multiple chunks for concurrent parallel I/O
/// 3. Consumed to extract the final data as a Vec<u8>
pub struct OwnedAlignedBuffer {
    backing: Arc<AlignedBacking>,
}

impl OwnedAlignedBuffer {
    /// Creates a new aligned buffer with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Size in bytes (should be block-aligned for best performance)
    pub fn new(capacity: usize) -> IoResult<Self> {
        Ok(Self {
            backing: Arc::new(AlignedBacking::new(capacity)?),
        })
    }

    /// Creates a view into a slice of this buffer.
    ///
    /// This is used for parallel I/O where multiple tasks write to different
    /// offsets of the same buffer concurrently.
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset
    /// * `len` - Length of the slice in bytes
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> IoResult<AlignedChunk> {
        AlignedChunk::new(Arc::clone(&self.backing), offset, len)
    }

    /// Consumes the buffer and returns the inner Vec<u8> with the specified length.
    ///
    /// This transfers ownership without copying data. The buffer must have no
    /// outstanding chunk references (all chunks must be dropped first).
    ///
    /// # Arguments
    ///
    /// * `len` - The actual data length (may be less than capacity due to padding)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Length exceeds capacity
    /// - There are still outstanding chunk references (Arc strong count > 1)
    pub fn into_vec(self, len: usize) -> IoResult<Vec<u8>> {
        let backing = Arc::try_unwrap(self.backing).map_err(|arc| {
            let refs = Arc::strong_count(&arc);
            IoError::other(format!(
                "cannot extract Vec: {refs} outstanding chunk reference(s) still exist"
            ))
        })?;
        backing.into_vec(len)
    }
}

// ---------------------------------------------------------------------------
// Direct I/O file operations
// ---------------------------------------------------------------------------

/// Open a file for direct reading (O_DIRECT) using tokio-uring.
#[inline]
pub async fn open_direct_read_io_uring(path: &Path) -> IoResult<UringFile> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .await
}

/// Open a file for direct writing (O_DIRECT) using tokio-uring.
#[inline]
pub async fn open_direct_write_io_uring(path: &Path) -> IoResult<UringFile> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .await
}

/// Open a file for direct reading (O_DIRECT) using Tokio's async file type.
#[inline]
pub async fn open_direct_read_tokio(path: &Path) -> IoResult<tokio::fs::File> {
    TokioOpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .await
}

/// Open a file for direct writing (O_DIRECT) using Tokio's async file type.
#[inline]
pub async fn open_direct_write_tokio(path: &Path) -> IoResult<tokio::fs::File> {
    TokioOpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .await
}

/// Open a file for direct reading (O_DIRECT) using std::fs.
#[inline]
pub fn open_direct_read_sync(path: &Path) -> IoResult<std::fs::File> {
    StdOpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

/// Open a file for direct writing (O_DIRECT) using std::fs.
#[inline]
pub fn open_direct_write_sync(path: &Path) -> IoResult<std::fs::File> {
    StdOpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_uring::buf::IoBuf;

    #[test]
    fn test_align_to_block() {
        assert_eq!(align_to_block(0), 0);
        assert_eq!(align_to_block(1), BLOCK_SIZE);
        assert_eq!(align_to_block(BLOCK_SIZE), BLOCK_SIZE);
        assert_eq!(align_to_block(BLOCK_SIZE + 1), BLOCK_SIZE * 2);
    }

    #[test]
    fn test_alloc_aligned_empty() {
        let buf = alloc_aligned(0).unwrap();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 0);
    }

    #[test]
    fn test_alloc_aligned_basic() {
        let capacity = BLOCK_SIZE * 2;
        let buf = alloc_aligned(capacity).unwrap();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), capacity);
        // Verify alignment - addr() gives us the address as usize
        assert_eq!(buf.as_ptr().addr() % BLOCK_SIZE, 0);
    }

    #[test]
    fn test_owned_aligned_buffer_basic() {
        let buffer = OwnedAlignedBuffer::new(BLOCK_SIZE).unwrap();
        let data_len = 100;
        let vec = buffer.into_vec(data_len).unwrap();
        assert_eq!(vec.len(), data_len);
    }

    #[test]
    fn test_owned_aligned_buffer_slice() {
        let buffer = OwnedAlignedBuffer::new(BLOCK_SIZE * 2).unwrap();
        let chunk1 = buffer.slice(0, BLOCK_SIZE).unwrap();
        let chunk2 = buffer.slice(BLOCK_SIZE, BLOCK_SIZE).unwrap();

        // Both chunks should be valid
        assert_eq!(chunk1.bytes_total(), BLOCK_SIZE);
        assert_eq!(chunk2.bytes_total(), BLOCK_SIZE);
    }

    #[test]
    fn test_owned_aligned_buffer_outstanding_refs() {
        let buffer = OwnedAlignedBuffer::new(BLOCK_SIZE).unwrap();
        let _chunk = buffer.slice(0, BLOCK_SIZE).unwrap();

        // Should fail because chunk still holds a reference
        let result = buffer.into_vec(100);
        assert!(result.is_err());
    }

    #[test]
    fn test_align_to_block_large_sizes() {
        assert_eq!(align_to_block(BLOCK_SIZE * 100), BLOCK_SIZE * 100);
        assert_eq!(align_to_block(BLOCK_SIZE * 100 + 1), BLOCK_SIZE * 101);
        assert_eq!(align_to_block(BLOCK_SIZE * 100 - 1), BLOCK_SIZE * 100);
    }

    #[test]
    fn test_is_block_aligned_edge_cases() {
        assert!(is_block_aligned(0, 0));
        assert!(is_block_aligned(0, BLOCK_SIZE));
        assert!(is_block_aligned(BLOCK_SIZE_U64, BLOCK_SIZE));
        assert!(!is_block_aligned(1, BLOCK_SIZE));
        assert!(!is_block_aligned(0, 1));
        assert!(!is_block_aligned(BLOCK_SIZE_U64, BLOCK_SIZE + 1));
    }

    #[test]
    fn test_can_use_direct_read_aligned() {
        assert!(can_use_direct_read(BLOCK_SIZE, BLOCK_SIZE));
        assert!(can_use_direct_read(BLOCK_SIZE * 2, BLOCK_SIZE));
        assert!(can_use_direct_read(BLOCK_SIZE * 4, BLOCK_SIZE * 2));
    }

    #[test]
    fn test_can_use_direct_read_unaligned() {
        assert!(!can_use_direct_read(0, BLOCK_SIZE));
        assert!(!can_use_direct_read(BLOCK_SIZE, BLOCK_SIZE + 1));
        assert!(!can_use_direct_read(BLOCK_SIZE + 1, BLOCK_SIZE));
        assert!(!can_use_direct_read(BLOCK_SIZE * 3, BLOCK_SIZE * 2));
    }

    #[test]
    fn test_can_use_direct_write_edge_cases() {
        assert!(can_use_direct_write(0));
        assert!(can_use_direct_write(BLOCK_SIZE));
        assert!(can_use_direct_write(BLOCK_SIZE * 100));
        assert!(!can_use_direct_write(1));
        assert!(!can_use_direct_write(BLOCK_SIZE - 1));
        assert!(!can_use_direct_write(BLOCK_SIZE + 1));
    }

    #[test]
    fn test_pad_to_block() {
        assert_eq!(pad_to_block(0), 0);
        assert_eq!(pad_to_block(1), BLOCK_SIZE);
        assert_eq!(pad_to_block(BLOCK_SIZE), BLOCK_SIZE);
        assert_eq!(pad_to_block(BLOCK_SIZE + 1), BLOCK_SIZE * 2);
    }

    #[test]
    fn test_alloc_aligned_large() {
        let capacity = BLOCK_SIZE * 1024;
        let buf = alloc_aligned(capacity).unwrap();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), capacity);
        assert_eq!(buf.as_ptr().addr() % BLOCK_SIZE, 0);
    }

    #[test]
    fn test_owned_aligned_buffer_into_vec_exceeds_capacity() {
        let buffer = OwnedAlignedBuffer::new(BLOCK_SIZE).unwrap();
        let result = buffer.into_vec(BLOCK_SIZE * 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_aligned_chunk_out_of_bounds() {
        let buffer = OwnedAlignedBuffer::new(BLOCK_SIZE).unwrap();
        let result = buffer.slice(0, BLOCK_SIZE * 2);
        assert!(result.is_err());
        let result = buffer.slice(BLOCK_SIZE, 1);
        assert!(result.is_err());
    }
}
