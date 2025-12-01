//! io_uring-based async I/O backend for Linux
//!
//! Uses tokio-uring for high-performance async file operations with true zero-copy.
//! - Uses buffer pools to minimize allocations
//! - Efficient parallel chunked reads for large files
//! - Proper io_uring batching - all operations submitted together

use super::odirect::{
    BLOCK_SIZE, OwnedAlignedBuffer, align_to_block, alloc_aligned, can_use_direct_read,
    can_use_direct_write, is_block_aligned, open_direct_read_io_uring, open_direct_write_io_uring,
};
use super::{
    IoResult,
    batch::{IndexedRequest, flatten_results, group_requests_by_file},
    get_buffer_pool,
};
use std::path::Path;
use tokio_uring::fs::File as UringFile;

/// Maximum size for a single io_uring read operation.
/// Files larger than this automatically use parallel chunked reading.
const MAX_SINGLE_READ: usize = 512 * 1024 * 1024; // 512MB

// ---------------------------------------------------------------------------
// Alignment helpers
// ---------------------------------------------------------------------------

#[inline]
fn statx_file_size(stat: libc::statx) -> IoResult<usize> {
    let size = stat.stx_size;
    usize::try_from(size).map_err(|_| std::io::Error::other("File too large"))
}

#[inline]
fn allow_direct_fallback(err: &std::io::Error) -> bool {
    matches!(
        err.raw_os_error(),
        Some(libc::EINVAL) | Some(libc::EOPNOTSUPP)
    )
}

// ---------------------------------------------------------------------------
// Validation helpers (pure functions)
// ---------------------------------------------------------------------------

/// Validates that the expected number of bytes were read.
#[inline]
fn validate_read_count(actual: usize, expected: usize) -> IoResult<()> {
    if actual < expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected to read {expected} bytes, but read {actual}"),
        ));
    }
    Ok(())
}

/// Validates that chunk size doesn't exceed Vec capacity limits.
fn validate_chunk_size(chunk_size: usize, file_size: usize) -> IoResult<()> {
    let max_capacity = usize::try_from(isize::MAX)
        .expect("isize::MAX should always fit in usize on the same platform");
    if chunk_size > max_capacity {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Chunk size ({chunk_size} bytes) exceeds maximum Vec capacity ({max_capacity} bytes). \
                 Increase chunks to at least {} to proceed.",
                file_size.div_ceil(max_capacity)
            ),
        ));
    }
    Ok(())
}

/// Helper for checked arithmetic operations with consistent error messages.
#[inline]
fn checked_arithmetic<T>(result: Option<T>, operation: &str) -> IoResult<T> {
    result.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("chunk calculation {operation}"),
        )
    })
}

#[derive(Clone, Copy)]
struct ChunkRequest {
    start: usize,
    len: usize,
    padded_len: usize,
    offset: u64,
}

fn build_chunk_requests(
    file_size: usize,
    chunk_size: usize,
    chunks: usize,
) -> IoResult<Vec<ChunkRequest>> {
    (0..chunks)
        .map(|i| {
            let start = checked_arithmetic(i.checked_mul(chunk_size), "overflow")?;

            if start >= file_size {
                return Ok(None);
            }

            let end = std::cmp::min(
                checked_arithmetic(start.checked_add(chunk_size), "overflow")?,
                file_size,
            );
            let len = checked_arithmetic(end.checked_sub(start), "underflow")?;

            if len == 0 {
                return Ok(None);
            }

            let offset = u64::try_from(start).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "offset too large")
            })?;
            let padded_len = align_to_block(len);

            Ok(Some(ChunkRequest {
                start,
                len,
                padded_len,
                offset,
            }))
        })
        .take_while(|result| matches!(result, Ok(Some(_)) | Err(_)))
        .map(|result| {
            result.and_then(|opt| {
                opt.ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "unexpected None in chunk request",
                    )
                })
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public read operations
// ---------------------------------------------------------------------------

/// Load an entire file using `io_uring`.
#[inline]
pub async fn load(path: impl AsRef<Path> + Send) -> IoResult<Vec<u8>> {
    let path_buf = path.as_ref().to_path_buf();
    let stat = tokio_uring::fs::statx(&path_buf).await?;
    let file_size = statx_file_size(stat)?;

    if file_size == 0 {
        return Ok(Vec::new());
    }

    if file_size > MAX_SINGLE_READ {
        let num_cpus = num_cpus::get();
        let min_chunks_for_size = file_size.div_ceil(MAX_SINGLE_READ);
        let chunks = std::cmp::max(num_cpus, min_chunks_for_size);
        return load_parallel(path_buf, chunks).await;
    }

    let padded = align_to_block(file_size);

    if can_use_direct_read(file_size, file_size) {
        match open_direct_read_io_uring(&path_buf).await {
            Ok(file) => {
                let buffer = OwnedAlignedBuffer::new(padded)?;
                let chunk = buffer.slice(0, padded)?;

                let (res, returned_chunk) = file.read_at(chunk, 0).await;
                let n = res?;

                validate_read_count(n, file_size)?;
                drop(returned_chunk);
                file.close().await?;
                return buffer.into_vec(file_size);
            }
            Err(err) if allow_direct_fallback(&err) => {}
            Err(err) => return Err(err),
        }
    }

    let file = UringFile::open(&path_buf).await?;
    let pooled = get_buffer_pool().get(file_size);
    let (res, mut buf) = file.read_at(pooled, 0).await;
    let n = res?;

    validate_read_count(n, file_size)?;
    buf.truncate(file_size);
    file.close().await?;
    Ok(buf.to_vec())
}

/// Load tensor data in parallel chunks using `io_uring` with batched operations.
#[inline]
pub async fn load_parallel(path: impl AsRef<Path>, chunks: usize) -> IoResult<Vec<u8>> {
    if chunks == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "chunks must be greater than 0",
        ));
    }

    let path_ref = path.as_ref();
    let stat = tokio_uring::fs::statx(path_ref).await?;
    let file_size = statx_file_size(stat)?;
    if file_size == 0 {
        return Ok(Vec::new());
    }

    let chunk_size = align_to_block(file_size.div_ceil(chunks));
    validate_chunk_size(chunk_size, file_size)?;

    let requests = build_chunk_requests(file_size, chunk_size, chunks)?;

    let required_capacity = requests
        .iter()
        .map(|r| r.start.saturating_add(r.padded_len))
        .max()
        .unwrap_or(file_size);
    let buffer = OwnedAlignedBuffer::new(align_to_block(required_capacity))?;

    if can_use_direct_read(file_size, chunk_size) {
        match open_direct_read_io_uring(path_ref).await {
            Ok(file) => {
                let file_ref = &file;
                let read_futures = requests
                    .iter()
                    .map(|req| {
                        let ChunkRequest {
                            start,
                            len: actual_chunk_size,
                            padded_len,
                            offset,
                        } = *req;
                        let chunk = buffer.slice(start, padded_len)?;
                        Ok(async move {
                            let (res, _chunk) = file_ref.read_at(chunk, offset).await;
                            let n = res?;
                            validate_read_count(n, actual_chunk_size)?;
                            IoResult::Ok(())
                        })
                    })
                    .collect::<IoResult<Vec<_>>>()?;

                let results = futures::future::join_all(read_futures).await;
                for res in results {
                    res?;
                }

                file.close().await?;
                return buffer.into_vec(file_size);
            }
            Err(err) if allow_direct_fallback(&err) => {}
            Err(err) => return Err(err),
        }
    }

    let file = UringFile::open(path_ref).await?;
    let file_ref = &file;

    let read_futures = requests
        .iter()
        .map(|req| {
            let ChunkRequest {
                start,
                len: actual_chunk_size,
                padded_len,
                offset,
            } = *req;
            let chunk = buffer.slice(start, padded_len)?;
            Ok(async move {
                let (res, _chunk) = file_ref.read_at(chunk, offset).await;
                let n = res?;
                validate_read_count(n, actual_chunk_size)?;
                IoResult::Ok(())
            })
        })
        .collect::<IoResult<Vec<_>>>()?;

    let results = futures::future::join_all(read_futures).await;
    for res in results {
        res?;
    }

    file.close().await?;
    buffer.into_vec(file_size)
}

/// Load a specific byte range from a file using `io_uring`.
#[inline]
pub async fn load_range(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    if len == 0 {
        return Ok(Vec::new());
    }

    if is_block_aligned(offset, len) {
        match open_direct_read_io_uring(path.as_ref()).await {
            Ok(file) => {
                let padded = align_to_block(len);
                let buffer = OwnedAlignedBuffer::new(padded)?;
                let chunk = buffer.slice(0, padded)?;

                let (res, returned_chunk) = file.read_at(chunk, offset).await;
                let n = res?;
                validate_read_count(n, len)?;
                drop(returned_chunk);
                file.close().await?;
                return buffer.into_vec(len);
            }
            Err(err) if allow_direct_fallback(&err) => {}
            Err(err) => return Err(err),
        }
    }

    let file = UringFile::open(path.as_ref()).await?;
    let pooled = get_buffer_pool().get(len);
    let (res, mut buf) = file.read_at(pooled, offset).await;
    let n = res?;
    validate_read_count(n, len)?;
    buf.truncate(len);
    file.close().await?;
    Ok(buf.to_vec())
}

/// Helpers for batched range reads.
async fn read_request_direct(
    file: &UringFile,
    req: &IndexedRequest,
) -> IoResult<(usize, Vec<u8>, usize, usize)> {
    let padded = align_to_block(req.len);
    let buffer = OwnedAlignedBuffer::new(padded)?;
    let chunk = buffer.slice(0, padded)?;

    let (res, returned_chunk) = file.read_at(chunk, req.offset).await;
    let n = res?;
    validate_read_count(n, req.len)?;
    drop(returned_chunk);
    let full = buffer.into_vec(req.len)?;
    Ok((req.idx, full, 0, req.len))
}

async fn read_request_buffered(
    file: &UringFile,
    req: &IndexedRequest,
) -> IoResult<(usize, Vec<u8>, usize, usize)> {
    let pooled = get_buffer_pool().get(req.len);
    let (res, mut read_buf) = file.read_at(pooled, req.offset).await;
    let n = res?;
    validate_read_count(n, req.len)?;
    read_buf.truncate(req.len);
    Ok((req.idx, read_buf.to_vec(), 0, req.len))
}

async fn process_file_batch(
    path: &Path,
    requests: Vec<IndexedRequest>,
) -> IoResult<Vec<(usize, Vec<u8>, usize, usize)>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let all_aligned = requests
        .iter()
        .all(|req| is_block_aligned(req.offset, req.len));
    let (file, use_direct) = if all_aligned {
        match open_direct_read_io_uring(path).await {
            Ok(file) => (file, true),
            Err(err) if allow_direct_fallback(&err) => (UringFile::open(path).await?, false),
            Err(err) => return Err(err),
        }
    } else {
        (UringFile::open(path).await?, false)
    };
    let file_ref = &file;

    let read_futures = requests.into_iter().map(|req| async move {
        if req.len == 0 {
            return Ok((req.idx, Vec::new(), 0, 0));
        }

        if use_direct {
            read_request_direct(file_ref, &req).await
        } else {
            read_request_buffered(file_ref, &req).await
        }
    });

    let results: IoResult<Vec<_>> = futures::future::join_all(read_futures)
        .await
        .into_iter()
        .collect();

    file.close().await?;
    results
}

/// Load multiple file ranges in a single batched operation.
#[inline]
pub async fn load_batch(
    requests: &[(impl AsRef<Path>, u64, usize)],
) -> IoResult<Vec<(Vec<u8>, usize, usize)>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let file_requests = group_requests_by_file(requests);

    let batch_results: Vec<Vec<super::batch::BatchResult>> = futures::future::join_all(
        file_requests
            .into_iter()
            .map(|(path, file_reqs)| async move { process_file_batch(&path, file_reqs).await }),
    )
    .await
    .into_iter()
    .collect::<IoResult<Vec<_>>>()?;

    Ok(flatten_results(batch_results))
}

// ---------------------------------------------------------------------------
// Public write operations
// ---------------------------------------------------------------------------

/// Write an entire buffer to a file, creating or truncating it first.
#[inline]
pub async fn write_all(path: impl AsRef<Path>, data: Vec<u8>) -> IoResult<()> {
    let path_ref = path.as_ref();
    if data.is_empty() {
        let file = open_direct_write_io_uring(path_ref).await?;
        file.sync_all().await?;
        return file.close().await;
    }

    let len = data.len();

    let write_with_file = |file: UringFile, buf: Vec<u8>, expected_len: usize| async move {
        let (res, written_buf) = file.write_at(buf, 0).submit().await;
        let n = res?;
        drop(written_buf);
        if n < expected_len {
            file.close().await?;
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                format!("expected to write {} bytes, wrote {}", expected_len, n),
            ));
        }
        file.sync_all().await?;
        file.close().await
    };

    if !can_use_direct_write(len) {
        let file = UringFile::create(path_ref).await?;
        return write_with_file(file, data, len).await;
    }

    let aligned_data = if data.as_ptr().align_offset(BLOCK_SIZE) == 0 {
        data
    } else {
        let mut buf = alloc_aligned(len)?;
        buf.extend_from_slice(&data);
        buf
    };

    let file = open_direct_write_io_uring(path_ref).await?;
    write_with_file(file, aligned_data, len).await
}
