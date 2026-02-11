//! Zero-copy buffer slicing for parallel I/O operations.
//!
//! This module provides safe abstractions for splitting a buffer into
//! non-overlapping mutable slices that can be passed across async tasks
//! for true zero-copy parallel I/O.
//!
//! # Safety Model
//!
//! The `BufferSlice` type provides a thread-safe wrapper around raw pointers
//! that enables zero-copy parallel I/O. Safety is guaranteed through:
//!
//! 1. **Non-overlapping slices**: Each `BufferSlice` represents a unique region
//! 2. **Lifetime management**: Parent buffer must outlive all `BufferSlice` instances
//! 3. **Exclusive access**: Each slice is accessed by exactly one task at a time
//! 4. **Const construction**: Zero-overhead abstraction with compile-time safety
//!
//! # Usage Example
//!
//! ```rust,ignore
//! // Pre-allocate final buffer
//! let mut final_buf = PooledBuffer::with_capacity(total_size);
//!
//! // Split into non-overlapping slices
//! let mut slices = Vec::new();
//! for i in 0..chunks {
//!     let start = i * chunk_size;
//!     let end = (start + chunk_size).min(total_size);
//!     let slice = final_buf.as_mut_slice().get_mut(start..end).unwrap();
//!
//!     // SAFETY: slice is non-overlapping and will be used exclusively
//!     let buffer_slice = unsafe { BufferSlice::from_slice(slice) };
//!     slices.push(buffer_slice);
//! }
//!
//! // Pass slices to parallel tasks
//! let handles = slices.into_iter().map(|mut slice| {
//!     tokio::spawn(async move {
//!         // SAFETY: exclusive access to this slice
//!         let data_slice = unsafe { slice.as_mut_slice() };
//!         // Read directly into data_slice (zero-copy)
//!         file.read_exact(data_slice).await
//!     })
//! });
//!
//! // Wait for all operations to complete
//! for handle in handles {
//!     handle.await??;
//! }
//! ```

/// A thread-safe wrapper around a mutable buffer slice for parallel I/O.
///
/// This struct enables zero-copy parallel I/O by allowing multiple async tasks
/// to write to non-overlapping regions of a pre-allocated buffer.
///
/// # Safety Guarantees
///
/// - Each `BufferSlice` represents a non-overlapping region of the parent buffer
/// - The parent buffer must remain valid for the lifetime of all `BufferSlice` instances
/// - Each slice is accessed by exactly one task at a time (no concurrent access)
/// - All slices must be consumed (via `as_mut_slice()`) before the parent buffer is used
pub struct BufferSlice {
    ptr: *mut u8,
    len: usize,
    _phantom: std::marker::PhantomData<&'static mut [u8]>,
}

// SAFETY: BufferSlice can be safely sent across threads because:
// 1. It represents exclusive ownership of a non-overlapping memory region
// 2. The raw pointer is only dereferenced via `as_mut_slice()` which requires `&mut self`
// 3. Callers must guarantee the backing buffer outlives all BufferSlice instances
// 4. The PhantomData marker ensures proper variance and drop checking
unsafe impl Send for BufferSlice {}

impl BufferSlice {
    /// Create a `BufferSlice` from a mutable slice.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of the following:
    ///
    /// 1. **Validity**: The backing memory remains valid and allocated for the
    ///    entire lifetime of this `BufferSlice`.
    ///
    /// 2. **Exclusivity**: No other code reads from or writes to this memory
    ///    region until this `BufferSlice` is dropped or fully consumed.
    ///
    /// 3. **Non-overlapping**: This slice does not overlap with any other
    ///    `BufferSlice` instances that may be accessed concurrently.
    ///
    /// 4. **Single-threaded access**: The resulting `BufferSlice` will only be
    ///    accessed by one task/thread at a time (though it can be moved between
    ///    threads via `Send`).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut buffer = vec![0u8; 100];
    /// let (left, right) = buffer.split_at_mut(50);
    ///
    /// // SAFETY: left and right are non-overlapping, and we maintain
    /// // exclusive access to each slice in separate tasks.
    /// let slice1 = unsafe { BufferSlice::from_slice(left) };
    /// let slice2 = unsafe { BufferSlice::from_slice(right) };
    /// ```
    pub const unsafe fn from_slice(slice: &mut [u8]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Reconstruct the mutable slice for reading/writing.
    ///
    /// # Safety
    ///
    /// The caller must guarantee:
    ///
    /// 1. **Validity**: The backing buffer is still allocated and valid.
    ///
    /// 2. **Exclusivity**: No other code is currently accessing this memory
    ///    region, and the returned slice will not be aliased.
    ///
    /// 3. **Non-overlapping**: No other `BufferSlice` pointing to overlapping
    ///    memory is being accessed concurrently.
    ///
    /// Note: Taking `&mut self` provides some protection against multiple
    /// simultaneous calls, but does not prevent the caller from creating
    /// multiple mutable references if they call this in a loop.
    pub const unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: Caller guarantees the pointer is valid, properly aligned,
        // and points to `self.len` initialized bytes. The `&mut self` receiver
        // ensures we have exclusive access to this BufferSlice.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Get the length of the slice.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the slice is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_slice_len() {
        let mut data = [0u8; 10];
        let slice = &mut data[2..7];

        // SAFETY: slice is valid and we're not accessing it elsewhere
        let buffer_slice = unsafe { BufferSlice::from_slice(slice) };

        assert_eq!(buffer_slice.len(), 5);
        assert!(!buffer_slice.is_empty());
    }

    #[test]
    fn test_buffer_slice_empty() {
        let mut data = [0u8; 5];
        let slice = &mut data[2..2]; // Empty slice

        // SAFETY: empty slice is always safe
        let buffer_slice = unsafe { BufferSlice::from_slice(slice) };

        assert_eq!(buffer_slice.len(), 0);
        assert!(buffer_slice.is_empty());
    }

    #[test]
    fn test_buffer_slice_roundtrip() {
        let mut data = [1u8, 2, 3, 4, 5];
        let slice = &mut data[1..4];

        // SAFETY: exclusive access to slice
        let mut buffer_slice = unsafe { BufferSlice::from_slice(slice) };

        // SAFETY: single access to reconstruct slice
        let reconstructed = unsafe { buffer_slice.as_mut_slice() };

        // Modify through reconstructed slice
        reconstructed[0] = 10;
        reconstructed[1] = 20;
        reconstructed[2] = 30;

        // Verify changes
        assert_eq!(data, [1, 10, 20, 30, 5]);
        assert_eq!(&data[1..4], &[10, 20, 30]);
    }

    #[test]
    fn test_buffer_slice_large_slice() {
        let mut data = vec![0u8; 1024 * 1024]; // 1MB
        let slice = &mut data[..];

        // SAFETY: exclusive access
        let buffer_slice = unsafe { BufferSlice::from_slice(slice) };

        assert_eq!(buffer_slice.len(), 1024 * 1024);
        assert!(!buffer_slice.is_empty());
    }

    #[test]
    fn test_buffer_slice_full_array() {
        let mut data = [42u8; 100];
        let slice = &mut data[..];

        // SAFETY: exclusive access
        let mut buffer_slice = unsafe { BufferSlice::from_slice(slice) };

        // SAFETY: reconstruct slice
        let reconstructed = unsafe { buffer_slice.as_mut_slice() };

        // Verify all elements
        assert_eq!(reconstructed.len(), 100);
        assert!(reconstructed.iter().all(|&x| x == 42));

        // Modify all
        for elem in reconstructed.iter_mut() {
            *elem = 99;
        }

        // Verify changes
        assert!(data.iter().all(|&x| x == 99));
    }

    #[test]
    fn test_buffer_slice_single_element() {
        let mut data = [7u8];
        let slice = &mut data[..];

        // SAFETY: exclusive access
        let mut buffer_slice = unsafe { BufferSlice::from_slice(slice) };

        assert_eq!(buffer_slice.len(), 1);

        // SAFETY: reconstruct
        let reconstructed = unsafe { buffer_slice.as_mut_slice() };
        assert_eq!(reconstructed[0], 7);

        reconstructed[0] = 13;
        assert_eq!(data[0], 13);
    }

    #[test]
    fn test_buffer_slice_non_overlapping() {
        let mut data = [0u8; 100];

        // Create two non-overlapping slices
        let (left, right) = data.split_at_mut(50);

        // SAFETY: slices are non-overlapping
        let slice1 = unsafe { BufferSlice::from_slice(left) };
        let slice2 = unsafe { BufferSlice::from_slice(right) };

        assert_eq!(slice1.len(), 50);
        assert_eq!(slice2.len(), 50);
    }

    #[test]
    fn test_buffer_slice_send() {
        // Verify BufferSlice can be sent across threads
        fn assert_send<T: Send>() {}
        assert_send::<BufferSlice>();
    }
}
