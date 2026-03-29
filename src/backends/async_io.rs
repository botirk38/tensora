//! Portable async I/O backend using Tokio.
//!
//! This backend provides cross-platform async I/O operations built on Tokio.
//! It's used as a fallback on non-Linux platforms where `io_uring` isn't available.

use super::{AsyncBackend, AsyncBackendFuture, BatchRequest, IoResult};
#[cfg(target_os = "linux")]
use super::{buffer_slice::BufferSlice, get_buffer_pool};
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::sync::Arc;
#[cfg(target_os = "linux")]
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};

/// Tokio-based async backend (cross-platform fallback).
#[derive(Clone, Copy, Debug)]
pub struct TokioAsyncBackend;

impl AsyncBackend for TokioAsyncBackend {
    fn load<'a>(&'a self, path: &'a Path) -> AsyncBackendFuture<'a, Vec<u8>> {
        Box::pin(async move { load(path).await })
    }

    fn load_parallel<'a>(
        &'a self,
        path: &'a Path,
        chunks: usize,
    ) -> AsyncBackendFuture<'a, Vec<u8>> {
        Box::pin(async move { load_parallel(path, chunks).await })
    }

    fn load_range<'a>(
        &'a self,
        path: &'a Path,
        offset: u64,
        len: usize,
    ) -> AsyncBackendFuture<'a, Vec<u8>> {
        Box::pin(async move { load_range(path, offset, len).await })
    }

    fn load_batch<'a>(
        &'a self,
        requests: &'a [BatchRequest],
    ) -> AsyncBackendFuture<'a, Vec<super::batch::FlattenedResult>> {
        Box::pin(async move {
            let owned: Vec<(PathBuf, u64, usize)> = requests
                .iter()
                .map(|(path, offset, len)| (path.clone(), *offset, *len))
                .collect();
            load_range_batch(&owned).await
        })
    }

    fn write_all<'a>(&'a self, path: &'a Path, data: Vec<u8>) -> AsyncBackendFuture<'a, ()> {
        Box::pin(async move { write_all(path, data).await })
    }
}

/// Ceiling division: (a + b - 1) / b
#[cfg(target_os = "linux")]
#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

#[cfg(target_os = "linux")]
mod linux {
    use super::super::batch::{flatten_results, group_requests_by_file};
    use super::super::odirect::{
        BLOCK_SIZE, alloc_aligned, can_use_direct_read, can_use_direct_write, is_block_aligned,
        open_direct_read_tokio, open_direct_write_tokio,
    };
    use super::*;
    use tokio::fs::File as TokioFile;
    use tokio::io::AsyncWriteExt;

    #[inline]
    fn allow_direct_fallback(err: &std::io::Error) -> bool {
        matches!(err.raw_os_error(), Some(libc::EINVAL | libc::EOPNOTSUPP))
    }

    /// Loads an entire file into memory using Direct I/O when possible.
    ///
    /// # Errors
    ///
    /// - File cannot be opened or read
    /// - File size exceeds `usize` limits
    #[inline]
    pub async fn load<P: AsRef<Path>>(path: P) -> IoResult<Vec<u8>> {
        let path_ref = path.as_ref();
        let mut file = TokioFile::open(path_ref).await?;
        let metadata = file.metadata().await?;
        let file_size = usize::try_from(metadata.len()).map_err(|_e| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "file too large")
        })?;

        if file_size == 0 {
            return Ok(Vec::new());
        }

        if can_use_direct_read(file_size, file_size) {
            match open_direct_read_tokio(path_ref).await {
                Ok(mut direct_file) => {
                    let mut buf = alloc_aligned(file_size)?;
                    buf.resize(file_size, 0);
                    direct_file.read_exact(&mut buf[..]).await?;
                    return Ok(buf);
                }
                Err(err) if allow_direct_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }

        let mut buf = get_buffer_pool().get(file_size);
        file.read_exact(buf.as_mut_slice()).await?;
        buf.truncate(file_size);
        Ok(buf.to_vec())
    }

    /// Internal write implementation using Direct I/O when possible.
    ///
    /// # Errors
    ///
    /// - File cannot be created or written to
    /// - Sync operation fails
    #[inline]
    pub async fn write_all_internal(path: &Path, data: Vec<u8>) -> IoResult<()> {
        if data.is_empty() {
            let file = open_direct_write_tokio(path).await?;
            file.sync_all().await?;
            return Ok(());
        }

        let len = data.len();

        if can_use_direct_write(len) {
            let is_aligned = data.as_ptr().align_offset(BLOCK_SIZE) == 0;
            if is_aligned {
                let mut file = open_direct_write_tokio(path).await?;
                file.write_all(&data).await?;
                file.sync_all().await
            } else {
                let mut buf = alloc_aligned(len)?;
                buf.extend_from_slice(&data);

                let mut file = open_direct_write_tokio(path).await?;
                file.write_all(&buf).await?;
                file.sync_all().await
            }
        } else {
            let mut file = TokioFile::create(path).await?;
            file.write_all(&data).await?;
            file.sync_all().await
        }
    }

    /// Loads a file using parallel chunk reads for improved throughput.
    ///
    /// # Errors
    ///
    /// - `chunks` is zero
    /// - File cannot be opened or read
    /// - File size exceeds `usize` limits
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
        let file_size = usize::try_from(metadata.len()).map_err(|_e| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "file too large")
        })?;

        let chunk_size = div_ceil(file_size, chunks);

        if can_use_direct_read(file_size, chunk_size) {
            match open_direct_read_tokio(path_ref).await {
                Ok(_) => {
                    drop(file);
                    let mut final_buf = alloc_aligned(file_size)?;
                    final_buf.resize(file_size, 0);
                    let path_buf = path_ref.to_path_buf();

                    let mut handles = Vec::with_capacity(chunks);
                    for i in 0..chunks {
                        let start = i * chunk_size;
                        let end = std::cmp::min(start + chunk_size, file_size);
                        if start >= end {
                            break;
                        }

                        let chunk_slice =
                            final_buf
                                .as_mut_slice()
                                .get_mut(start..end)
                                .ok_or_else(|| {
                                    std::io::Error::new(
                                        std::io::ErrorKind::InvalidInput,
                                        "invalid chunk range",
                                    )
                                })?;

                        let mut buffer_slice = unsafe { BufferSlice::from_slice(chunk_slice) };
                        let path_clone = path_buf.clone();

                        let handle = tokio::spawn(async move {
                            let mut direct_file =
                                open_direct_read_tokio(path_clone.as_path()).await?;
                            let start_offset = u64::try_from(start).map_err(|_e| {
                                std::io::Error::new(
                                    std::io::ErrorKind::InvalidInput,
                                    "seek offset too large",
                                )
                            })?;
                            direct_file.seek(SeekFrom::Start(start_offset)).await?;
                            let slice = unsafe { buffer_slice.as_mut_slice() };
                            direct_file.read_exact(slice).await?;
                            IoResult::Ok(())
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.await??;
                    }

                    return Ok(final_buf);
                }
                Err(err) if allow_direct_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }

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

        let mut final_buf = get_buffer_pool().get(file_size);
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

            let chunk_slice = final_buf
                .as_mut_slice()
                .get_mut(start..end)
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid chunk range")
                })?;

            let mut buffer_slice = unsafe { BufferSlice::from_slice(chunk_slice) };
            let path_clone = path_ref.to_path_buf();

            let handle = tokio::spawn(async move {
                // Open a NEW file handle instead of cloning to avoid shared file position cursor
                let mut file_for_chunk = TokioFile::open(&path_clone).await?;

                file_for_chunk
                    .seek(SeekFrom::Start(u64::try_from(start).map_err(|_e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "seek offset too large",
                        )
                    })?))
                    .await?;

                let slice = unsafe { buffer_slice.as_mut_slice() };
                file_for_chunk.read_exact(slice).await?;
                IoResult::Ok(())
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await??;
        }

        final_buf.truncate(file_size);
        Ok(final_buf.to_vec())
    }

    /// Loads a range of bytes from a file at the specified offset.
    ///
    /// # Errors
    ///
    /// - File cannot be opened or read
    /// - Seek or read operation fails
    #[inline]
    pub async fn load_range<P: AsRef<Path>>(path: P, offset: u64, len: usize) -> IoResult<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }

        if is_block_aligned(offset, len) {
            match open_direct_read_tokio(path.as_ref()).await {
                Ok(mut file) => {
                    let mut buf = alloc_aligned(len)?;
                    buf.resize(len, 0);
                    file.seek(SeekFrom::Start(offset)).await?;
                    file.read_exact(buf.as_mut_slice()).await?;
                    return Ok(buf);
                }
                Err(err) if allow_direct_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }

        let mut file = TokioFile::open(path.as_ref()).await?;
        file.seek(SeekFrom::Start(offset)).await?;
        let mut buf = get_buffer_pool().get(len);
        file.read_exact(buf.as_mut_slice()).await?;
        buf.truncate(len);
        Ok(buf.to_vec())
    }

    #[inline]
    pub async fn load_range_batch(
        requests: &[(impl AsRef<Path> + Send + Sync, u64, usize)],
    ) -> IoResult<Vec<(Arc<[u8]>, usize, usize)>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);

        let file_futures = grouped.into_iter().map(|(path, reqs)| async move {
            let read_futures = reqs.into_iter().map(|req| {
                let path_ref = path.clone();
                async move {
                    let data = load_range(&path_ref, req.offset, req.len).await?;
                    let owned: Arc<[u8]> = data.into();
                    IoResult::Ok((req.idx, owned, 0, req.len))
                }
            });

            futures::future::join_all(read_futures)
                .await
                .into_iter()
                .collect::<IoResult<Vec<_>>>()
        });

        let results = futures::future::join_all(file_futures)
            .await
            .into_iter()
            .collect::<IoResult<Vec<_>>>()?;

        Ok(flatten_results(results))
    }
}

#[cfg(not(target_os = "linux"))]
mod non_linux {
    use super::super::batch::group_requests_by_file;
    use super::*;
    use std::sync::Arc;

    type IndexedLoadResult = (usize, Arc<[u8]>, usize, usize);
    use tokio::fs::File as TokioFile;

    /// Loads an entire file into memory.
    ///
    /// On non-Linux platforms, regular files do not support true async I/O.
    /// This delegates to the sync backend via `spawn_blocking` to avoid
    /// tokio::fs overhead (double indirection through blocking pool).
    ///
    /// # Errors
    ///
    /// - File cannot be opened or read
    /// - File size exceeds `usize` limits
    #[inline]
    pub async fn load<P: AsRef<Path>>(path: P) -> IoResult<Vec<u8>> {
        let path_buf = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || super::super::sync_backend().load(path_buf.as_path()))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    /// Loads a file in parallel chunks using the sync backend.
    ///
    /// On non-Linux platforms, this delegates to the sync backend's
    /// `load_parallel` (native threads + blocking I/O) via `spawn_blocking`,
    /// matching sync parallel performance while preserving the async API.
    ///
    /// # Errors
    ///
    /// - `chunks` is zero
    /// - File cannot be opened or read
    /// - File size exceeds `usize` limits
    #[inline]
    pub async fn load_parallel<P: AsRef<Path>>(path: P, chunks: usize) -> IoResult<Vec<u8>> {
        if chunks == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "chunks must be greater than 0",
            ));
        }
        let path_buf = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || {
            super::super::sync_backend().load_parallel(path_buf.as_path(), chunks)
        })
        .await
        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    #[inline]
    pub async fn load_range<P: AsRef<Path>>(path: P, offset: u64, len: usize) -> IoResult<Vec<u8>> {
        let path_buf = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || {
            super::super::sync_backend().load_range(path_buf.as_path(), offset, len)
        })
        .await
        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    #[inline]
    pub async fn load_range_batch(
        requests: &[(impl AsRef<Path> + Send + Sync, u64, usize)],
    ) -> IoResult<Vec<(Arc<[u8]>, usize, usize)>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);

        let file_futures: Vec<_> = grouped
            .into_iter()
            .map(|(path, indexed_reqs)| {
                let path_buf = path.clone();
                async move {
                    tokio::task::spawn_blocking(move || -> Result<Vec<IndexedLoadResult>, std::io::Error> {
                        use std::io::{Seek, SeekFrom, Read};
                        let mut file = std::fs::File::open(&path_buf)?;
                        let mut results = Vec::with_capacity(indexed_reqs.len());
                        let mut sorted_reqs = indexed_reqs;
                        sorted_reqs.sort_by_key(|r| r.offset);

                        for req in sorted_reqs {
                            file.seek(SeekFrom::Start(req.offset))?;
                            let mut buf = super::super::get_buffer_pool().get(req.len);
                            file.read_exact(&mut buf[..])?;
                            let owned: Arc<[u8]> = buf.into_inner().into();
                            results.push((req.idx, owned, 0, req.len));
                        }

                        results.sort_by_key(|(idx, _, _, _)| *idx);
                        Ok(results)
                    })
                    .await
                    .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
                }
            })
            .collect();

        let file_results = futures::future::join_all(file_futures)
            .await
            .into_iter()
            .collect::<IoResult<Vec<_>>>()?;

        let mut indexed: Vec<IndexedLoadResult> = file_results.into_iter().flatten().collect();
        indexed.sort_by_key(|(idx, _, _, _)| *idx);

        Ok(indexed
            .into_iter()
            .map(|(_, data, _, len)| (data, 0, len))
            .collect())
    }

    #[inline]
    pub async fn write_all_internal(path: &Path, data: Vec<u8>) -> IoResult<()> {
        use tokio::io::AsyncWriteExt;

        if data.is_empty() {
            let file = TokioFile::create(path).await?;
            file.sync_all().await?;
            return Ok(());
        }

        let mut file = TokioFile::create(path).await?;
        file.write_all(&data).await?;
        file.sync_all().await
    }
}

/// Write an entire buffer to a file asynchronously.
///
/// # Errors
///
/// - File cannot be created or written to
/// - Sync operation fails
#[inline]
pub async fn write_all<P: AsRef<Path>>(path: P, data: Vec<u8>) -> IoResult<()> {
    #[cfg(target_os = "linux")]
    {
        linux::write_all_internal(path.as_ref(), data).await
    }
    #[cfg(not(target_os = "linux"))]
    {
        non_linux::write_all_internal(path.as_ref(), data).await
    }
}

#[cfg(target_os = "linux")]
pub use linux::*;

#[cfg(not(target_os = "linux"))]
pub use non_linux::*;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // -----------------------------------------------------------------------
    // Unit Tests - Pure Functions
    // -----------------------------------------------------------------------

    #[cfg(target_os = "linux")]
    mod helpers {
        use super::*;

        #[test]
        fn test_div_ceil_basic() {
            assert_eq!(div_ceil(10, 3), 4);
            assert_eq!(div_ceil(9, 3), 3);
            assert_eq!(div_ceil(1, 3), 1);
        }

        #[test]
        fn test_div_ceil_exact_division() {
            assert_eq!(div_ceil(12, 4), 3);
            assert_eq!(div_ceil(100, 10), 10);
        }

        #[test]
        fn test_div_ceil_zero() {
            assert_eq!(div_ceil(0, 5), 0);
        }

        #[test]
        fn test_div_ceil_large_numbers() {
            assert_eq!(div_ceil(1_000_000, 3), 333_334);
        }
    }

    // -----------------------------------------------------------------------
    // Integration Tests - Cross-Platform
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_load_empty_file() {
        let tmpfile = NamedTempFile::new().unwrap();
        let result = load(tmpfile.path()).await.unwrap();
        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_load_small_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"test data").unwrap();
        tmpfile.flush().unwrap();

        let result = load(tmpfile.path()).await.unwrap();
        assert_eq!(result, b"test data");
    }

    #[tokio::test]
    async fn test_load_larger_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = vec![0xAB; 1024 * 1024]; // 1MB
        tmpfile.write_all(&data).unwrap();
        tmpfile.flush().unwrap();

        let result = load(tmpfile.path()).await.unwrap();
        assert_eq!(result, data);
    }

    // Note: load_parallel with small files can be flaky on tmpfs
    // This is tested more reliably in sync_io.rs and io_uring.rs
    // and in the test_load_parallel_vs_sequential test below

    #[tokio::test]
    async fn test_load_parallel_single_chunk() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = vec![0xEF; 1024 * 5];
        tmpfile.write_all(&data).unwrap();
        tmpfile.flush().unwrap();

        let result = load_parallel(tmpfile.path(), 1).await.unwrap();
        assert_eq!(result, data);
    }

    #[tokio::test]
    async fn test_load_parallel_zero_chunks_error() {
        let tmpfile = NamedTempFile::new().unwrap();
        let result = load_parallel(tmpfile.path(), 0).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[tokio::test]
    async fn test_load_parallel_empty_file() {
        let tmpfile = NamedTempFile::new().unwrap();
        let result = load_parallel(tmpfile.path(), 4).await.unwrap();
        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_load_parallel_vs_sequential() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = vec![0x42; 1024 * 500]; // 500KB - large enough for parallel
        tmpfile.write_all(&data).unwrap();
        tmpfile.as_file().sync_all().unwrap(); // Ensure data is written
        tmpfile.flush().unwrap();

        let sequential = load(tmpfile.path()).await.unwrap();
        let parallel = load_parallel(tmpfile.path(), 8).await.unwrap();

        assert_eq!(sequential, parallel);
        assert_eq!(sequential, data);
    }

    #[tokio::test]
    async fn test_load_range_basic() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        tmpfile.write_all(data).unwrap();
        tmpfile.flush().unwrap();

        let result = load_range(tmpfile.path(), 5, 10).await.unwrap();
        assert_eq!(result, &data[5..15]);
    }

    #[tokio::test]
    async fn test_load_range_full_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = b"complete file content";
        tmpfile.write_all(data).unwrap();
        tmpfile.flush().unwrap();

        let result = load_range(tmpfile.path(), 0, data.len()).await.unwrap();
        assert_eq!(result, data);
    }

    #[tokio::test]
    async fn test_load_range_zero_length() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"test").unwrap();
        tmpfile.flush().unwrap();

        let result = load_range(tmpfile.path(), 0, 0).await.unwrap();
        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_write_all_basic() {
        let tmpfile = NamedTempFile::new().unwrap();
        let data = b"Hello, async_io!".to_vec();

        write_all(tmpfile.path(), data.clone()).await.unwrap();

        // Read back and verify
        let result = load(tmpfile.path()).await.unwrap();
        assert_eq!(result, data);
    }

    #[tokio::test]
    async fn test_write_all_empty() {
        let tmpfile = NamedTempFile::new().unwrap();
        write_all(tmpfile.path(), Vec::new()).await.unwrap();

        let result = load(tmpfile.path()).await.unwrap();
        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_write_all_large() {
        let tmpfile = NamedTempFile::new().unwrap();
        let data = vec![0x55; 1024 * 1024]; // 1MB

        write_all(tmpfile.path(), data.clone()).await.unwrap();

        let result = load(tmpfile.path()).await.unwrap();
        assert_eq!(result, data);
    }

    #[tokio::test]
    async fn test_load_range_batch_single_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        tmpfile.write_all(data).unwrap();
        tmpfile.flush().unwrap();

        let requests = vec![
            (tmpfile.path(), 0, 10),
            (tmpfile.path(), 10, 10),
            (tmpfile.path(), 20, 10),
        ];

        let results = load_range_batch(&requests).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0.as_ref(), &data[0..10]);
        assert_eq!(results[1].0.as_ref(), &data[10..20]);
        assert_eq!(results[2].0.as_ref(), &data[20..30]);
    }

    #[tokio::test]
    async fn test_load_range_batch_multiple_files() {
        let mut tmpfile1 = NamedTempFile::new().unwrap();
        let mut tmpfile2 = NamedTempFile::new().unwrap();
        tmpfile1.write_all(b"FILE1DATA").unwrap();
        tmpfile1.flush().unwrap();
        tmpfile2.write_all(b"FILE2DATA").unwrap();
        tmpfile2.flush().unwrap();

        let requests = vec![
            (tmpfile1.path(), 0, 5),
            (tmpfile2.path(), 0, 5),
            (tmpfile1.path(), 5, 4),
        ];

        let results = load_range_batch(&requests).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0.as_ref(), b"FILE1");
        assert_eq!(results[1].0.as_ref(), b"FILE2");
        assert_eq!(results[2].0.as_ref(), b"DATA");
    }

    #[tokio::test]
    async fn test_load_range_batch_empty() {
        let requests: Vec<(&str, u64, usize)> = vec![];
        let results = load_range_batch(&requests).await.unwrap();
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_load_missing_file() {
        let result = load("/nonexistent/file/path").await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Linux-Specific Tests
    // -----------------------------------------------------------------------

    #[cfg(target_os = "linux")]
    mod linux_tests {
        use super::*;

        #[tokio::test]
        async fn test_load_with_direct_io_attempt() {
            // This test verifies that O_DIRECT path is attempted
            // It may fall back to buffered I/O depending on filesystem
            let mut tmpfile = NamedTempFile::new().unwrap();
            let data = vec![0x77; 4096 * 2]; // Block-aligned size
            tmpfile.write_all(&data).unwrap();
            tmpfile.flush().unwrap();

            let result = load(tmpfile.path()).await.unwrap();
            assert_eq!(result, data);
        }

        #[tokio::test]
        async fn test_load_parallel_direct_io() {
            let mut tmpfile = NamedTempFile::new().unwrap();
            let data = vec![0x88; 4096 * 100]; // Block-aligned, large enough for parallel
            tmpfile.write_all(&data).unwrap();
            tmpfile.as_file().sync_all().unwrap(); // Ensure data is written
            tmpfile.flush().unwrap();

            let result = load_parallel(tmpfile.path(), 4).await.unwrap();
            assert_eq!(result, data);
        }

        #[tokio::test]
        async fn test_write_all_aligned_data() {
            let tmpfile = NamedTempFile::new().unwrap();
            let data = vec![0x99; 4096]; // Block-aligned

            write_all(tmpfile.path(), data.clone()).await.unwrap();

            let result = load(tmpfile.path()).await.unwrap();
            assert_eq!(result, data);
        }
    }
}
