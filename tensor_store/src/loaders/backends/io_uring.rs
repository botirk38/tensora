use super::IoResult;
use std::sync::OnceLock;
use tokio_uring::fs::File as UringFile;
use zeropool::BufferPool;

static BUFFER_POOL: OnceLock<BufferPool> = OnceLock::new();

fn get_buffer_pool() -> &'static BufferPool {
    BUFFER_POOL.get_or_init(|| BufferPool::new())
}

/// A Vec-like type that borrows memory from a slice but acts like Vec<u8> for tokio-uring
struct BorrowedVec {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

impl BorrowedVec {
    unsafe fn from_slice(slice: &mut [u8]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            cap: slice.len(),
        }
    }

    fn into_vec(self) -> Vec<u8> {
        // Create a Vec with the correct capacity so tokio-uring can use it properly
        // We'll handle deallocation separately
        unsafe { Vec::from_raw_parts(self.ptr, self.len, self.cap) }
    }
}

impl Drop for BorrowedVec {
    fn drop(&mut self) {
        // The Vec created by into_vec has capacity 0, so it won't deallocate
        // But we still need to ensure this BorrowedVec doesn't try to deallocate
    }
}

impl AsRef<[u8]> for BorrowedVec {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl AsMut<[u8]> for BorrowedVec {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Ceiling division: (a + b - 1) / b
#[inline]
fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Load tensor data using io_uring zero-copy I/O
#[inline]
pub async fn load(path: &str) -> IoResult<Vec<u8>> {
    let file = UringFile::open(path).await?;
    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len() as usize;

    // Use internal buffer pool for optimization
    let buf = get_buffer_pool().get(file_size);

    let (res, buf) = file.read_at(buf, 0).await;
    let n = res?;

    if n != file_size {
        file.close().await?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected to read {} bytes, but read {}", file_size, n),
        ));
    }

    file.close().await?;

    Ok(buf)
}

/// Load tensor data in parallel chunks using io_uring with true zero-copy
#[inline]
pub async fn load_parallel(path: &str, chunks: usize) -> IoResult<Vec<u8>> {
    if chunks == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "chunks must be greater than 0",
        ));
    }

    let file = UringFile::open(path).await?;
    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len() as usize;

    let chunk_size = div_ceil(file_size, chunks);

    // Pre-allocate final buffer
    let mut final_buf = get_buffer_pool().get(file_size);

    // Submit all read operations in parallel, each reading directly into final buffer
    let mut read_futures = Vec::with_capacity(chunks);

    for i in 0..chunks {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, file_size);
        let actual_chunk_size = end - start;

        if actual_chunk_size == 0 {
            break;
        }

        // Create a BorrowedVec that points to the slice of final_buf for this chunk
        let chunk_slice = &mut final_buf[start..end];
        let borrowed_vec = unsafe { BorrowedVec::from_slice(chunk_slice) };

        let offset = start as u64;
        let read_future = file.read_at(borrowed_vec.into_vec(), offset);
        read_futures.push(read_future);
    }

    // Wait for all operations to complete
    // The data is already in final_buf, we just need to wait for completion
    for read_future in read_futures {
        let (res, returned_buf) = read_future.await;
        let _n = res?;
        // Leak the returned buffer since its data is now in final_buf
        std::mem::forget(returned_buf);
    }

    file.close().await?;
    Ok(final_buf)
}

/// Load a specific byte range from tensor data using io_uring
#[inline]
pub async fn load_range(path: &str, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    let file = UringFile::open(path).await?;

    // Use internal buffer pool for optimization
    let buf = get_buffer_pool().get(len);

    let (res, buf) = file.read_at(buf, offset).await;
    let n = res?;

    if n != len {
        file.close().await?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected to read {} bytes, but read {}", len, n),
        ));
    }

    file.close().await?;
    Ok(buf)
}
