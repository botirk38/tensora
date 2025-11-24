//! io_uring-based async I/O backend for Linux
//!
//! Uses tokio-uring for high-performance async file operations with true zero-copy.
//! - Uses buffer pools to minimize allocations
//! - Efficient parallel chunked reads for large files
//! - Proper io_uring batching - all operations submitted together

use super::{IoResult, get_buffer_pool};
use std::path::Path;
use tokio_uring::fs::File as UringFile;

/// Load an entire file using `io_uring`.
///
/// Uses pooled buffers to minimize allocations and provides zero-copy reads.
///
/// # Arguments
///
/// * `path` - Path to the file to load
///
/// # Returns
///
/// Returns the file contents as a `Vec<u8>`.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
///
/// For files larger than 512MB, automatically uses parallel chunked reading
/// to work around io_uring single-read size limitations.
#[inline]
pub async fn load(path: impl AsRef<Path> + Send) -> IoResult<Vec<u8>> {
    let path_buf = path.as_ref().to_path_buf();
    let metadata = std::fs::metadata(&path_buf)?;
    let file_size =
        usize::try_from(metadata.len()).map_err(|_e| std::io::Error::other("File too large"))?;

    // io_uring read_at has practical limits on single-read size (typically ~700MB).
    // For larger files, automatically use parallel chunked reading.
    const MAX_SINGLE_READ: usize = 512 * 1024 * 1024; // 512MB threshold

    if file_size > MAX_SINGLE_READ {
        // Calculate optimal number of chunks:
        // - Prefer one chunk per CPU core for maximum parallelism
        // - But ensure each chunk doesn't exceed MAX_SINGLE_READ
        let num_cpus = num_cpus::get();
        let min_chunks_for_size = file_size.div_ceil(MAX_SINGLE_READ);
        let chunks = std::cmp::max(num_cpus, min_chunks_for_size);
        return load_parallel(path_buf, chunks).await;
    }

    // Small file - use single read
    let file = UringFile::open(&path_buf).await?;

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
    Ok(returned_buf.into_inner())
}

/// Load tensor data in parallel chunks using `io_uring` with batched operations.
///
/// Opens the file once and submits all chunk reads to io_uring in a single batch,
/// allowing the kernel to handle them concurrently.
///
/// # Arguments
///
/// * `path` - Path to the file to load
/// * `chunks` - Number of chunks to split the file into
///
/// # Returns
///
/// Returns the file contents as a `Vec<u8>`.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, read, or if chunk size exceeds limits.
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

    let chunk_size = file_size.div_ceil(chunks);

    // Validate chunk size doesn't exceed Vec capacity limit (isize::MAX)
    let max_capacity = usize::try_from(isize::MAX)
        .expect("isize::MAX should always fit in usize on the same platform");
    if chunk_size > max_capacity {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Chunk size ({} bytes) exceeds maximum Vec capacity ({} bytes). \
                 Increase chunks to at least {} to proceed.",
                chunk_size,
                max_capacity,
                file_size.div_ceil(max_capacity)
            ),
        ));
    }

    // Pre-allocate final pooled buffer - this is the ONLY allocation
    let mut final_buf = get_buffer_pool().get(file_size);

    // Open file ONCE for all operations - key difference from old implementation
    let file = UringFile::open(path_ref).await?;

    // SAFETY: We split final_buf into non-overlapping mutable slices.
    // Each slice is passed to exactly one read operation.
    // The buffer remains valid until all operations complete (we await all futures).
    let mut read_futures = Vec::with_capacity(chunks);

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

        let offset = u64::try_from(start).map_err(|_e| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "offset too large")
        })?;

        // Create non-overlapping mutable slice
        let chunk_slice = final_buf
            .as_mut_slice()
            .get_mut(start..end)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid chunk range")
            })?;

        // SAFETY: This slice is unique to this read operation and won't be accessed elsewhere.
        // We create a Vec from the raw parts to satisfy read_at's interface,
        // but the data is written directly to the pre-allocated buffer slice (ZERO COPY!)
        let slice_ptr = chunk_slice.as_mut_ptr();
        let slice_len = chunk_slice.len();
        let vec = unsafe { Vec::from_raw_parts(slice_ptr, 0, slice_len) };

        // Submit read operation - all will be batched by io_uring
        let read_future = file.read_at(vec, offset);
        read_futures.push((read_future, actual_chunk_size));
    }

    // Wait for all read operations to complete concurrently
    // io_uring handles these in parallel at the kernel level
    let results =
        futures::future::join_all(read_futures.into_iter().map(|(fut, expected)| async move {
            let (res, returned_vec) = fut.await;
            (res, returned_vec, expected)
        }))
        .await;

    // Check for errors and validate read sizes
    for (res, returned_vec, expected_size) in results {
        // Check for I/O errors
        let n = match res {
            Ok(n) => n,
            Err(e) => {
                // Don't drop returned_vec to avoid double-free
                std::mem::forget(returned_vec);
                return Err(e);
            }
        };

        // Validate read size
        if n != expected_size {
            // Don't drop returned_vec to avoid double-free
            std::mem::forget(returned_vec);
            file.close().await?;
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("Expected to read {expected_size} bytes, but read {n}"),
            ));
        }

        // Don't drop returned_vec - data is in final_buf slice
        std::mem::forget(returned_vec);
    }

    file.close().await?;

    Ok(final_buf.into_inner())
}

/// Load a specific byte range from a file using `io_uring`.
///
/// Uses pooled buffers to minimize allocations.
#[inline]
pub async fn load_range(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    let path_ref = path.as_ref();
    let file = UringFile::open(path_ref).await?;

    // Get buffer from pool (buffer pool optimization for reduced allocations)
    let pool = get_buffer_pool();
    let buf = pool.get(len);

    let (res, mut returned_buf) = file.read_at(buf, offset).await;
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
    returned_buf.truncate(n);
    Ok(returned_buf.into_inner())
}

/// Load multiple file ranges in a single batched operation.
///
/// All reads are submitted to io_uring concurrently for maximum throughput.
///
/// # Arguments
///
/// * `requests` - Slice of (path, offset, length) tuples
///
/// # Returns
///
/// Returns a vector of byte vectors, one per request, in the same order.
///
/// # Errors
///
/// Returns an error if any file cannot be opened or read.
#[inline]
pub async fn load_batch(requests: &[(impl AsRef<Path>, u64, usize)]) -> IoResult<Vec<Vec<u8>>> {
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
