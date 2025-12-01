//! Portable async I/O backend using Tokio.
//!
//! This backend provides cross-platform async I/O operations built on Tokio.
//! It's used as a fallback on non-Linux platforms where `io_uring` isn't available.

use super::{IoResult, buffer_slice::BufferSlice, get_buffer_pool};
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};

/// Ceiling division: (a + b - 1) / b
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
        matches!(
            err.raw_os_error(),
            Some(libc::EINVAL) | Some(libc::EOPNOTSUPP)
        )
    }

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
            let mut file_clone = file.try_clone().await?;

            let handle = tokio::spawn(async move {
                file_clone
                    .seek(SeekFrom::Start(u64::try_from(start).map_err(|_e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "seek offset too large",
                        )
                    })?))
                    .await?;

                let slice = unsafe { buffer_slice.as_mut_slice() };
                file_clone.read_exact(slice).await?;
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
    ) -> IoResult<Vec<(Vec<u8>, usize, usize)>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);

        let file_futures = grouped.into_iter().map(|(path, reqs)| async move {
            let read_futures = reqs.into_iter().map(|req| {
                let path_ref = path.clone();
                async move {
                    let data = load_range(&path_ref, req.offset, req.len).await?;
                    IoResult::Ok((req.idx, data, 0, req.len))
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
    use super::super::batch::{flatten_results, group_requests_by_file};
    use super::*;
    use tokio::fs::File as TokioFile;

    #[inline]
    pub async fn load<P: AsRef<Path>>(path: P) -> IoResult<Vec<u8>> {
        let path_ref = path.as_ref();
        let mut file = TokioFile::open(path_ref).await?;
        let metadata = file.metadata().await?;
        let file_size = usize::try_from(metadata.len()).map_err(|_e| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "file too large")
        })?;

        let mut buf = get_buffer_pool().get(file_size);
        file.read_exact(buf.as_mut_slice()).await?;
        buf.truncate(file_size);
        Ok(buf.to_vec())
    }

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
            let mut file_clone = file.try_clone().await?;

            let handle = tokio::spawn(async move {
                file_clone
                    .seek(SeekFrom::Start(u64::try_from(start).map_err(|_e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "seek offset too large",
                        )
                    })?))
                    .await?;

                let slice = unsafe { buffer_slice.as_mut_slice() };
                file_clone.read_exact(slice).await?;
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

    #[inline]
    pub async fn load_range<P: AsRef<Path>>(path: P, offset: u64, len: usize) -> IoResult<Vec<u8>> {
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
    ) -> IoResult<Vec<(Vec<u8>, usize, usize)>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);

        let file_futures = grouped.into_iter().map(|(path, reqs)| async move {
            let read_futures = reqs.into_iter().map(|req| {
                let path_ref = path.clone();
                async move {
                    let data = load_range(&path_ref, req.offset, req.len).await?;
                    IoResult::Ok((req.idx, data, 0, req.len))
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

    #[inline]
    pub async fn write_all_internal(path: &Path, data: Vec<u8>) -> IoResult<()> {
        use tokio::io::AsyncWriteExt;

        if data.is_empty() {
            let mut file = TokioFile::create(path).await?;
            file.sync_all().await?;
            return Ok(());
        }

        let mut file = TokioFile::create(path).await?;
        file.write_all(&data).await?;
        file.sync_all().await
    }
}

/// Write an entire buffer to a file asynchronously.
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
