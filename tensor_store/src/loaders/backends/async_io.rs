use super::IoResult;
use std::sync::OnceLock;
use tokio::fs::File as TokioFile;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};
use zeropool::BufferPool;

static BUFFER_POOL: OnceLock<BufferPool> = OnceLock::new();

fn get_buffer_pool() -> &'static BufferPool {
    BUFFER_POOL.get_or_init(|| BufferPool::new())
}

/// Helper struct to safely pass buffer slices to async tasks
///
/// SAFETY: This struct holds a raw pointer to a slice of a pre-allocated buffer.
/// The parent function retains ownership of the buffer and ensures:
/// 1. The buffer lives for the entire duration of all async tasks
/// 2. Each task gets a non-overlapping slice (no data races)
/// 3. The buffer is not moved or deallocated until all tasks complete
struct BufferSlice {
    ptr: *mut u8,
    len: usize,
    _phantom: std::marker::PhantomData<&'static mut [u8]>,
}

// SAFETY: We ensure non-overlapping slices across tasks
unsafe impl Send for BufferSlice {}

impl BufferSlice {
    /// Create a BufferSlice from a mutable slice
    ///
    /// SAFETY: Caller must ensure:
    /// - The slice remains valid for the lifetime of this BufferSlice
    /// - No other code accesses this slice until BufferSlice is consumed
    unsafe fn from_slice(slice: &mut [u8]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Reconstruct the mutable slice for reading
    ///
    /// SAFETY: This can only be called once per BufferSlice
    unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Ceiling division: (a + b - 1) / b
#[inline]
fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Load tensor data using portable async I/O
#[inline]
pub async fn load(path: &str) -> IoResult<Vec<u8>> {
    let mut file = TokioFile::open(path).await?;
    let metadata = file.metadata().await?;
    let file_size = metadata.len() as usize;

    // Use internal buffer pool for optimization
    let mut buf = get_buffer_pool().get(file_size);
    file.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Load tensor data in parallel chunks using portable async I/O with zero-copy
#[inline]
pub async fn load_parallel(path: &str, chunks: usize) -> IoResult<Vec<u8>> {
    if chunks == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "chunks must be greater than 0",
        ));
    }

    let file = TokioFile::open(path).await?;
    let metadata = file.metadata().await?;
    let file_size = metadata.len() as usize;

    let chunk_size = div_ceil(file_size, chunks);

    // Pre-allocate final buffer - this is the ONLY allocation
    let mut final_buf = get_buffer_pool().get(file_size);

    // SAFETY: We split final_buf into non-overlapping mutable slices.
    // Each slice is passed to exactly one task via BufferSlice.
    // The buffer remains valid until all tasks complete (we await all handles).
    // No other code accesses final_buf until all tasks are done.
    let mut handles = Vec::with_capacity(chunks);

    for i in 0..chunks {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, file_size);
        let actual_chunk_size = end - start;

        if actual_chunk_size == 0 {
            break;
        }

        // Create non-overlapping mutable slice
        let chunk_slice = &mut final_buf[start..end];

        // SAFETY: This slice is unique to this task and won't be accessed elsewhere
        let mut buffer_slice = unsafe { BufferSlice::from_slice(chunk_slice) };

        let mut file_clone = file.try_clone().await?;
        let handle = tokio::spawn(async move {
            // Seek to the correct position
            file_clone.seek(SeekFrom::Start(start as u64)).await?;

            // SAFETY: We're the only task with access to this BufferSlice
            let slice = unsafe { buffer_slice.as_mut_slice() };

            // Read directly into the final buffer slice (ZERO COPY!)
            file_clone.read_exact(slice).await?;

            IoResult::Ok(())
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await??;
    }

    // All data is now in final_buf, return it
    Ok(final_buf)
}

/// Load a specific byte range from tensor data using portable async I/O
#[inline]
pub async fn load_range(path: &str, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    let mut file = TokioFile::open(path).await?;

    // Seek to the specified offset
    file.seek(SeekFrom::Start(offset)).await?;

    // Use internal buffer pool for optimization
    let mut buf = get_buffer_pool().get(len);
    file.read_exact(&mut buf).await?;
    Ok(buf)
}
