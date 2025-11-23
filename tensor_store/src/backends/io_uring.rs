//! io_uring-based async I/O backend for Linux.
//!
//! This backend leverages Linux's `io_uring` interface for maximum performance.
//! It provides zero-copy operations and efficient parallel I/O for large files.
//!
//! # Platform Support
//!
//! This module is only available on Linux (requires kernel 5.1+).
//!
//! # Performance Characteristics
//!
//! - **Zero-copy**: Direct kernel-to-userspace transfers
//! - **Parallel I/O**: True concurrent reads via `io_uring` submission queue
//! - **Buffer pooling**: Reuses memory allocations across operations
//!
//! # Usage
//!
//! Typically accessed via `backends::load()` on Linux platforms, not used directly.

use super::{IoResult, buffer_slice::BufferSlice, get_buffer_pool};
use std::path::Path;
use tokio_uring::fs::File as UringFile;

/// Ceiling division: (a + b - 1) / b
#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Load tensor data using `io_uring` zero-copy I/O
#[inline]
pub async fn load(path: impl AsRef<Path> + Send) -> IoResult<Vec<u8>> {
    let path_buf = path.as_ref().to_path_buf();
    let file = UringFile::open(&path_buf).await?;
    let metadata = std::fs::metadata(&path_buf)?;
    let file_size =
        usize::try_from(metadata.len()).map_err(|_e| std::io::Error::other("File too large"))?;

    // Get buffer from pool
    let pool = get_buffer_pool();
    let buf = pool.get(file_size);

    let (res, returned_buf) = file.read_at(buf, 0).await;
    let n = res?;

    if n != file_size {
        file.close().await?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected to read {file_size} bytes, but read {n}"),
        ));
    }

    file.close().await?;

    // Buffer will be automatically returned to pool when dropped
    Ok(returned_buf.into_vec())
}

/// Load tensor data in parallel chunks using `io_uring` with true zero-copy
#[inline]
pub async fn load_parallel(path: impl AsRef<Path>, chunks: usize) -> IoResult<Vec<u8>> {
    if chunks == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "chunks must be greater than 0",
        ));
    }

    let path_ref = path.as_ref();
    let metadata = std::fs::metadata(path_ref)?;
    let file_size =
        usize::try_from(metadata.len()).map_err(|_e| std::io::Error::other("File too large"))?;

    let chunk_size = div_ceil(file_size, chunks);

    // Pre-allocate final pooled buffer - this is the ONLY allocation
    let mut final_buf = get_buffer_pool().get(file_size);

    // SAFETY: We split final_buf into non-overlapping mutable slices.
    // Each slice is passed to exactly one task via BufferSlice.
    // The buffer remains valid until all tasks complete (we await all futures).
    // No other code accesses final_buf until all futures are done.
    let mut futures = Vec::with_capacity(chunks);

    for i in 0..chunks {
        let start = i.checked_mul(chunk_size).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "chunk calculation overflow",
            )
        })?;
        let end = std::cmp::min(
            start.checked_add(chunk_size).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "chunk calculation overflow",
                )
            })?,
            file_size,
        );
        let actual_chunk_size = end.checked_sub(start).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "chunk calculation underflow",
            )
        })?;

        if actual_chunk_size == 0 {
            break;
        }

        // Create non-overlapping mutable slice
        let chunk_slice = final_buf
            .as_mut_slice()
            .get_mut(start..end)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid chunk range")
            })?;

        // SAFETY: This slice is unique to this future and won't be accessed elsewhere
        let mut buffer_slice = unsafe { BufferSlice::from_slice(chunk_slice) };

        let path_ref_clone = path_ref.to_path_buf();
        let future = async move {
            let file_clone = UringFile::open(&path_ref_clone).await?;
            let offset = u64::try_from(start).map_err(|_e| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "offset too large")
            })?;

            // SAFETY: We're the only future with access to this BufferSlice.
            // We create a Vec from the raw parts to satisfy read_at's interface,
            // but the data is written directly to the pre-allocated buffer slice (ZERO COPY!)
            let slice = unsafe { buffer_slice.as_mut_slice() };
            let vec = unsafe { Vec::from_raw_parts(slice.as_mut_ptr(), 0, slice.len()) };
            let (res, returned_vec) = file_clone.read_at(vec, offset).await;
            let n = res?;

            if n != actual_chunk_size {
                // Don't drop returned_vec to avoid double-free
                std::mem::forget(returned_vec);
                file_clone.close().await?;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("Expected to read {actual_chunk_size} bytes, but read {n}"),
                ));
            }

            // Don't drop returned_vec, data is in slice
            std::mem::forget(returned_vec);

            file_clone.close().await?;
            IoResult::Ok(())
        };

        futures.push(future);
    }

    // Wait for all futures to complete concurrently
    let results = futures::future::join_all(futures).await;

    // Check for errors
    for result in results {
        result?;
    }

    // All data is now in final_buf, set correct length
    final_buf.truncate(file_size);
    Ok(final_buf.into_inner())
}

/// Load a specific byte range from a file using `io_uring`.
#[inline]
pub async fn load_range(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    let path_ref = path.as_ref();
    let file = UringFile::open(path_ref).await?;

    // Get buffer from pool (buffer pool optimization for reduced allocations)
    let pool = get_buffer_pool();
    let buf = pool.get(len);
    let (res, mut read_buf) = file.read_at(buf, offset).await;
    let n = res?;

    if n != len {
        file.close().await?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected to read {len} bytes, but read {n}"),
        ));
    }

    file.close().await?;

    // Buffer will be automatically returned to pool when dropped
    read_buf.truncate(n);
    Ok(read_buf.into_inner())
}

/// Load multiple byte ranges from potentially different files using `io_uring` batching.
///
/// This function groups requests by file path, opens each file once, and submits
/// all read operations to the io_uring submission queue before awaiting any completions.
/// This provides true parallel I/O with multiple in-flight operations.
///
/// # Arguments
///
/// * `requests` - A slice of (path, offset, len) tuples specifying what to read
///
/// # Returns
///
/// A vector of `Vec<u8>` in the same order as the input requests.
///
/// # Errors
///
/// Returns an error if any file cannot be opened or if any read operation fails.
#[inline]
pub async fn load_range_batch(
    requests: &[(impl AsRef<Path> + Send + Sync, u64, usize)],
) -> IoResult<Vec<Vec<u8>>> {
    use std::collections::HashMap;

    if requests.is_empty() {
        return Ok(Vec::new());
    }

    // Group requests by file path to minimize file opens
    let mut file_requests: HashMap<std::path::PathBuf, Vec<(usize, u64, usize)>> = HashMap::new();

    for (idx, (path, offset, len)) in requests.iter().enumerate() {
        let path_buf = path.as_ref().to_path_buf();
        file_requests
            .entry(path_buf)
            .or_default()
            .push((idx, *offset, *len));
    }

    // Pre-allocate result vector
    let mut results = vec![None; requests.len()];

    // Open all files in parallel first
    let mut file_handles = Vec::with_capacity(file_requests.len());
    let mut open_futures = Vec::with_capacity(file_requests.len());

    for path in file_requests.keys() {
        let open_future = UringFile::open(path);
        open_futures.push(open_future);
    }

    let opened_files = futures::future::join_all(open_futures).await;
    for result in opened_files {
        file_handles.push(result?);
    }

    // Process each file with true parallel I/O batching
    for (file_idx, (_path, file_reqs)) in file_requests.into_iter().enumerate() {
        let file = &file_handles[file_idx];

        // Submit all read operations concurrently to io_uring submission queue
        // Uses pooled buffers for efficient memory management
        let mut read_futures = Vec::with_capacity(file_reqs.len());
        for (_idx, offset, len) in &file_reqs {
            let buf = get_buffer_pool().get(*len);
            let read_future = file.read_at(buf, *offset);
            read_futures.push(read_future);
        }

        // Wait for all operations to complete in parallel using join_all
        let file_results = futures::future::join_all(read_futures).await;

        // Process results and check for errors
        let mut processed_results = Vec::with_capacity(file_reqs.len());
        for (i, (res, mut read_buf)) in file_results.into_iter().enumerate() {
            let n = res?;
            if n != file_reqs[i].2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("Expected to read {} bytes, but read {}", file_reqs[i].2, n),
                ));
            }
            // Buffer will be automatically returned to pool when dropped
            read_buf.truncate(n);
            processed_results.push(read_buf.into_inner());
        }

        // Store results in the correct positions
        for ((idx, _offset, _len), data) in file_reqs.into_iter().zip(processed_results) {
            results[idx] = Some(data);
        }
    }

    // Close all files
    for file in file_handles {
        file.close().await?;
    }

    // Convert results to Vec, maintaining order
    let final_results: Vec<Vec<u8>> = results
        .into_iter()
        .map(|opt| opt.expect("All batch requests should have been processed"))
        .collect();

    Ok(final_results)
}

/// Write an entire buffer to a file, creating or truncating it first.
#[inline]
pub async fn write_all(path: impl AsRef<Path>, data: &[u8]) -> IoResult<()> {
    let path_ref = path.as_ref();
    let file = UringFile::create(path_ref).await?;
    let (res, buf) = file.write_at(data.to_vec(), 0).submit().await;
    let n = res?;
    if n != data.len() {
        file.close().await?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::WriteZero,
            format!("expected to write {} bytes, wrote {}", data.len(), n),
        ));
    }
    // Keep buf alive until write completes; then drop.
    drop(buf);
    file.sync_all().await?;
    file.close().await
}
