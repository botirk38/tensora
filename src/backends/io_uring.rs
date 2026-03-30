//! io_uring-based explicit I/O backend for Linux.
//!
//! This backend provides specialized Linux I/O using the `io_uring` interface directly,
//! as opposed to the default `AsyncReader`/`AsyncWriter` which are Tokio-backed on all
//! platforms.
//!
//! Use this directly when you want explicit io_uring behavior on Linux. On other
//! platforms this module is not available.
//!
//! # Example
//!
//! ```ignore
//! use tensor_store::backends::io_uring::Reader;
//!
//! let mut reader = Reader::new();
//! let data = reader.load("model.bin").await?;
//! ```

use super::odirect::{
    OwnedAlignedBuffer, align_to_block, can_use_direct_read,
    is_block_aligned, open_direct_read_io_uring,
};
use super::{
    BatchRequest, IoResult,
    batch::FlattenedResult,
    get_buffer_pool,
};
use std::path::Path;
use std::sync::Arc;
use tokio_uring::fs::File as UringFile;

const ASYNC_QUEUE_DEPTH: usize = 64;
const MIN_CHUNK_SIZE: usize = 32 * 1024 * 1024;
const MAX_SINGLE_READ: usize = 512 * 1024 * 1024;
const MAX_CHUNK_SIZE: usize = 256 * 1024 * 1024;

#[inline]
fn calculate_async_chunks(file_size: usize) -> usize {
    if file_size == 0 {
        return 1;
    }
    let size_based = if file_size > MAX_SINGLE_READ {
        file_size.div_ceil(64 * 1024 * 1024)
    } else {
        1
    };
    let queue_chunks = ASYNC_QUEUE_DEPTH.min(file_size.div_ceil(MIN_CHUNK_SIZE));
    size_based.max(queue_chunks).clamp(1, ASYNC_QUEUE_DEPTH)
}

#[inline]
fn statx_file_size(stat: libc::statx) -> IoResult<usize> {
    usize::try_from(stat.stx_size).map_err(|_| std::io::Error::other("File too large"))
}

#[inline]
fn allow_direct_fallback(err: &std::io::Error) -> bool {
    matches!(err.raw_os_error(), Some(libc::EINVAL | libc::EOPNOTSUPP))
}

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

fn validate_chunk_size(chunk_size: usize, file_size: usize) -> IoResult<()> {
    let max_capacity = usize::try_from(isize::MAX)
        .expect("isize::MAX should always fit in usize on the same platform");
    if chunk_size > max_capacity {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Chunk size ({chunk_size} bytes) exceeds maximum Vec capacity ({max_capacity} bytes). Increase chunks to at least {} to proceed.",
                file_size.div_ceil(max_capacity)
            ),
        ));
    }
    Ok(())
}

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
            Ok(Some(ChunkRequest {
                start,
                len,
                padded_len: align_to_block(len),
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

pub struct Reader;

impl Reader {
    pub const fn new() -> Self {
        Self
    }

    pub async fn load(&mut self, path: impl AsRef<Path> + Send) -> IoResult<Vec<u8>> {
        let path_buf = path.as_ref().to_path_buf();
        let stat = tokio_uring::fs::statx(&path_buf).await?;
        let file_size = statx_file_size(stat)?;
        if file_size == 0 {
            return Ok(Vec::new());
        }
        self.load_chunked(path_buf, file_size, calculate_async_chunks(file_size)).await
    }

    async fn load_chunked(&mut self, path: std::path::PathBuf, file_size: usize, chunks: usize) -> IoResult<Vec<u8>> {
        if chunks == 0 {
            return Err(std::io::Error::other("chunks must be > 0"));
        }
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
            match open_direct_read_io_uring(&path).await {
                Ok(file) => {
                    let file_ref = &file;
                    let read_futures = requests
                        .iter()
                        .map(|req| {
                            let ChunkRequest { start, len, padded_len, offset } = *req;
                            let chunk = buffer.slice(start, padded_len)?;
                            Ok(async move {
                                let (res, _chunk) = file_ref.read_at(chunk, offset).await;
                                validate_read_count(res?, len)
                            })
                        })
                        .collect::<IoResult<Vec<_>>>()?;

                    for res in futures::future::join_all(read_futures).await {
                        res?;
                    }
                    file.close().await?;
                    return buffer.into_vec(file_size);
                }
                Err(err) if allow_direct_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }

        let file = UringFile::open(&path).await?;
        let file_ref = &file;
        let read_futures = requests
            .iter()
            .map(|req| {
                let ChunkRequest { start, len, padded_len, offset } = *req;
                let chunk = buffer.slice(start, padded_len)?;
                Ok(async move {
                    let (res, _chunk) = file_ref.read_at(chunk, offset).await;
                    validate_read_count(res?, len)
                })
            })
            .collect::<IoResult<Vec<_>>>()?;

        for res in futures::future::join_all(read_futures).await {
            res?;
        }
        file.close().await?;
        buffer.into_vec(file_size)
    }

    pub async fn load_range(&mut self, path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
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
                    validate_read_count(res?, len)?;
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
        validate_read_count(res?, len)?;
        buf.truncate(len);
        file.close().await?;
        Ok(buf.into_inner())
    }

    pub async fn load_range_batch(&mut self, requests: &[BatchRequest]) -> IoResult<Vec<FlattenedResult>> {
        use super::batch::group_requests_by_file;

        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);
        let mut all_group_futures: Vec<_> = Vec::with_capacity(grouped.len());

        for (path, reqs) in grouped {
            if reqs.is_empty() {
                continue;
            }

            let path_clone = path.clone();
            let reqs_clone: Vec<_> = reqs.to_vec();

            let group_future = async move {
                let mut results: Vec<(usize, Arc<[u8]>, usize, usize)> =
                    Vec::with_capacity(reqs_clone.len());

                let file = UringFile::open(&path_clone).await
                    .map_err(std::io::Error::other)?;

                let mut pending: Vec<(usize, _, usize)> = Vec::new();

                for req in reqs_clone {
                    let offset = req.offset;
                    let len = req.len;
                    let idx = req.idx;

                    if len == 0 {
                        results.push((idx, Arc::new([]), 0, 0));
                        continue;
                    }

                    let pooled = get_buffer_pool().get(len);
                    let future = file.read_at(pooled, offset);
                    pending.push((idx, future, len));
                }

                for (idx, future, len) in pending {
                    let (res, mut buf) = future.await;
                    validate_read_count(res.map_err(std::io::Error::other)?, len)?;
                    buf.truncate(len);
                    let data: Vec<u8> = buf.into_inner();
                    results.push((idx, data.into(), 0, len));
                }

                let _ = file.close().await;
                ReaderResult::Ok(results)
            };

            all_group_futures.push(group_future);
        }

        let grouped_results = futures::future::join_all(all_group_futures)
                .await
                .into_iter()
                .collect::<Result<Vec<Vec<(usize, Arc<[u8]>, usize, usize)>>, std::io::Error>>()?;

        let mut indexed: Vec<(usize, Arc<[u8]>, usize, usize)> = grouped_results
            .into_iter()
            .flatten()
            .collect();
        indexed.sort_by_key(|(idx, _, _, _)| *idx);
        Ok(indexed.into_iter().map(|(_, data, offset, len)| (data, offset, len)).collect())
    }
}

impl Default for Reader {
    fn default() -> Self {
        Self::new()
    }
}

type ReaderResult<T> = Result<T, std::io::Error>;

pub struct Writer {
    file: Option<UringFile>,
    path: std::path::PathBuf,
}

impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer").finish_non_exhaustive()
    }
}

impl Writer {
    pub async fn create(path: &Path) -> IoResult<Self> {
        if let Some(parent) = path.parent() && !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = UringFile::create(path).await?;
        Ok(Self { file: Some(file), path: path.to_path_buf() })
    }

    pub async fn write_all(&mut self, data: Vec<u8>) -> IoResult<()> {
        let file = self.file.take().ok_or_else(|| std::io::Error::other("writer closed"))?;
        let _ = file.close().await;
        
        // Truncate using std::fs
        std::fs::File::create(&self.path).map_err(std::io::Error::other)?.set_len(0).map_err(std::io::Error::other)?;
        
        let file = UringFile::create(&self.path).await?;
        
        let mut offset = 0;
        while offset < data.len() {
            let this_chunk = std::cmp::min(MAX_CHUNK_SIZE, data.len() - offset);
            let chunk = data[offset..offset + this_chunk].to_vec();
            let (res, _) = file.write_at(chunk, offset as u64).submit().await;
            let n = res?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "write returned zero bytes"));
            }
            offset += n;
        }
        
        file.sync_all().await?;
        self.file = Some(file);
        Ok(())
    }

    pub async fn write_at(&mut self, offset: u64, data: Vec<u8>) -> IoResult<()> {
        let file = self.file.as_mut().ok_or_else(|| std::io::Error::other("writer closed"))?;
        
        let mut written = 0usize;
        while written < data.len() {
            let this_chunk = std::cmp::min(MAX_CHUNK_SIZE, data.len() - written);
            let chunk = data[written..written + this_chunk].to_vec();
            let (res, _) = file.write_at(chunk, offset + written as u64).submit().await;
            let n = res?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "write returned zero bytes"));
            }
            written += n;
        }
        
        Ok(())
    }

    pub async fn sync_all(&mut self) -> IoResult<()> {
        let file = self.file.as_ref().ok_or_else(|| std::io::Error::other("writer closed"))?;
        file.sync_all().await
    }
}
