//! Portable async I/O backend using Tokio.
//!
//! This backend provides cross-platform async I/O operations built on Tokio.
//! It's used as a fallback on non-Linux platforms where `io_uring` isn't available.
//!
//! # Platform Support
//!
//! Works on all platforms supported by Tokio (Linux, macOS, Windows, etc.).
//!
//! # Performance Characteristics
//!
//! - **Zero-copy parallel reads**: Direct writes to final buffer via unsafe slicing
//! - **Async I/O**: Non-blocking operations via Tokio runtime
//! - **Buffer pooling**: Reuses memory allocations across operations
//!
//! # Usage
//!
//! Typically accessed via `backends::load()` on non-Linux platforms, not used directly.

use super::{IoResult, buffer_slice::BufferSlice, get_buffer_pool};
use std::path::Path;
#[allow(unused_imports)]
use tokio::fs::File as TokioFile;
#[allow(unused_imports)]
use tokio::io::AsyncWriteExt;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};

/// Ceiling division: (a + b - 1) / b
#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Load tensor data using portable async I/O
#[inline]
pub async fn load<P: AsRef<Path>>(path: P) -> IoResult<Vec<u8>> {
    let path_ref = path.as_ref();
    let mut file = TokioFile::open(path_ref).await?;
    let metadata = file.metadata().await?;
    let file_size = usize::try_from(metadata.len())
        .map_err(|_e| std::io::Error::new(std::io::ErrorKind::InvalidInput, "file too large"))?;

    // Use pooled buffer for optimization
    let mut buf = get_buffer_pool().get(file_size);
    file.read_exact(buf.as_mut_slice()).await?;
    buf.truncate(file_size);
    Ok(buf.into_vec())
}

/// Load tensor data in parallel chunks using portable async I/O with zero-copy
#[inline]
pub async fn load_parallel<P: AsRef<Path>>(path: P, chunks: usize) -> IoResult<Vec<u8>> {
    if chunks == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "chunks must be greater than 0",
        ));
    }

    let path_ref = path.as_ref();
    let file = TokioFile::open(path_ref).await?;
    let metadata = file.metadata().await?;
    let file_size = usize::try_from(metadata.len())
        .map_err(|_e| std::io::Error::new(std::io::ErrorKind::InvalidInput, "file too large"))?;

    let chunk_size = div_ceil(file_size, chunks);

    // Pre-allocate final pooled buffer - this is the ONLY allocation
    // (zero-copy: tasks write directly into non-overlapping buffer slices)
    let mut final_buf = get_buffer_pool().get(file_size);

    // SAFETY: We split final_buf into non-overlapping mutable slices.
    // Each slice is passed to exactly one task via BufferSlice.
    // The buffer remains valid until all tasks complete (we await all handles).
    // No other code accesses final_buf until all tasks are done.
    let mut handles = Vec::with_capacity(chunks);

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

        // SAFETY: This slice is unique to this task and won't be accessed elsewhere
        let mut buffer_slice = unsafe { BufferSlice::from_slice(chunk_slice) };

        let mut file_clone = file.try_clone().await?;
        let handle = tokio::spawn(async move {
            // Seek to the correct position
            file_clone
                .seek(SeekFrom::Start(u64::try_from(start).map_err(|_e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek offset too large")
                })?))
                .await?;

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

    // All data is now in final_buf, set correct length
    final_buf.truncate(file_size);
    Ok(final_buf.into_inner())
}

/// Load a specific byte range from a file asynchronously.
#[inline]
pub async fn load_range<P: AsRef<Path>>(path: P, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    let mut file = TokioFile::open(path).await?;
    file.seek(SeekFrom::Start(offset)).await?;
    let mut buf = get_buffer_pool().get(len);
    file.read_exact(buf.as_mut_slice()).await?;
    buf.truncate(len);
    Ok(buf.into_vec())
}

/// Load multiple byte ranges from files asynchronously.
///
/// This function processes multiple range requests concurrently, grouping by file path
/// for efficiency. Each request is a tuple of (path, offset, len).
///
/// # Arguments
///
/// * `requests` - Slice of (path, offset, len) tuples to load
///
/// # Returns
///
/// A vector of byte vectors in the same order as the requests.
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

    // Process each file with concurrent range requests
    for (_path, file_reqs) in file_requests {
        // Submit all range read operations concurrently
        let mut read_futures = Vec::with_capacity(file_reqs.len());

        for (_idx, offset, len) in &file_reqs {
            let read_future = load_range(&_path, *offset, *len);
            read_futures.push(read_future);
        }

        // Wait for all operations to complete in parallel
        let file_results = futures::future::join_all(read_futures).await;

        // Process results and check for errors
        let mut processed_results = Vec::with_capacity(file_reqs.len());
        for result in file_results {
            processed_results.push(result?);
        }

        // Store results in the correct positions
        for ((idx, _offset, _len), data) in file_reqs.into_iter().zip(processed_results) {
            results[idx] = Some(data);
        }
    }

    // Convert results to Vec, maintaining order
    let final_results: Vec<Vec<u8>> = results
        .into_iter()
        .map(|opt| opt.expect("All batch requests should have been processed"))
        .collect();

    Ok(final_results)
}
