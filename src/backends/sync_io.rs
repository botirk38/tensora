//! Synchronous blocking I/O using std::fs

use super::{IoResult, buffer_slice::BufferSlice, get_buffer_pool};
use std::path::Path;

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
        open_direct_read_sync, open_direct_write_sync,
    };
    use super::*;
    use std::fs::File;
    use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
    use std::thread;

    #[inline]
    fn allow_direct_fallback(err: &std::io::Error) -> bool {
        matches!(
            err.raw_os_error(),
            Some(libc::EINVAL) | Some(libc::EOPNOTSUPP)
        )
    }

    pub fn load(path: impl AsRef<Path>) -> IoResult<Vec<u8>> {
        let mut file = File::open(path.as_ref())?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_foo| Error::new(ErrorKind::InvalidInput, "file too large"))?;

        if len == 0 {
            return Ok(Vec::new());
        }

        if can_use_direct_read(len, len) {
            match open_direct_read_sync(path.as_ref()) {
                Ok(mut direct_file) => {
                    let mut buf = alloc_aligned(len)?;
                    buf.resize(len, 0);
                    direct_file.read_exact(&mut buf[..])?;
                    return Ok(buf);
                }
                Err(err) if allow_direct_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }

        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(buf.into_inner())
    }

    #[inline]
    pub fn load_parallel<P: AsRef<Path>>(path: P, chunks: usize) -> IoResult<Vec<u8>> {
        if chunks == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "chunks must be greater than 0",
            ));
        }

        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let metadata = file.metadata()?;
        let file_size = usize::try_from(metadata.len()).map_err(|_e| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "file too large")
        })?;

        let chunk_size = div_ceil(file_size, chunks);

        if can_use_direct_read(file_size, chunk_size) {
            match open_direct_read_sync(path_ref) {
                Ok(_) => {
                    drop(file);
                    let mut final_buf = alloc_aligned(file_size)?;
                    final_buf.resize(file_size, 0);

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
                        let path_buf = path_ref.to_path_buf();

                        let handle = thread::spawn(move || {
                            let mut direct_file = open_direct_read_sync(path_buf.as_path())?;
                            direct_file.seek(SeekFrom::Start(u64::try_from(start).map_err(
                                |_e| {
                                    std::io::Error::new(
                                        std::io::ErrorKind::InvalidInput,
                                        "seek offset too large",
                                    )
                                },
                            )?))?;

                            let slice = unsafe { buffer_slice.as_mut_slice() };
                            direct_file.read_exact(slice)?;
                            IoResult::Ok(())
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        handle
                            .join()
                            .map_err(|_e| std::io::Error::other("thread panicked"))??;
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
            let mut file_clone = file.try_clone()?;

            let handle = thread::spawn(move || {
                file_clone.seek(SeekFrom::Start(u64::try_from(start).map_err(|_e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek offset too large")
                })?))?;

                let slice = unsafe { buffer_slice.as_mut_slice() };
                file_clone.read_exact(slice)?;
                IoResult::Ok(())
            });

            handles.push(handle);
        }

        for handle in handles {
            handle
                .join()
                .map_err(|_e| std::io::Error::other("thread panicked"))??;
        }

        final_buf.truncate(file_size);
        Ok(final_buf.into_inner())
    }

    pub fn load_range(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }

        if is_block_aligned(offset, len) {
            match open_direct_read_sync(path.as_ref()) {
                Ok(mut file) => {
                    let mut buf = alloc_aligned(len)?;
                    buf.resize(len, 0);
                    file.seek(SeekFrom::Start(offset))?;
                    file.read_exact(&mut buf[..])?;
                    return Ok(buf);
                }
                Err(err) if allow_direct_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }

        let mut file = File::open(path.as_ref())?;
        file.seek(SeekFrom::Start(offset))?;

        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(buf.into_inner())
    }

    pub fn load_range_batch(
        requests: &[(impl AsRef<Path> + Send + Sync, u64, usize)],
    ) -> IoResult<Vec<super::super::batch::FlattenedResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);

        let mut results = Vec::with_capacity(requests.len());
        for (path, reqs) in grouped {
            let mut handles = Vec::with_capacity(reqs.len());

            for req in reqs {
                let path_clone = path.clone();
                handles.push(thread::spawn(move || {
                    let data = load_range(&path_clone, req.offset, req.len)?;
                    IoResult::Ok((req.idx, data, 0, req.len))
                }));
            }

            let mut file_results = Vec::with_capacity(handles.len());
            for handle in handles {
                file_results.push(
                    handle
                        .join()
                        .map_err(|_e| std::io::Error::other("thread panicked"))??,
                );
            }

            results.push(file_results);
        }

        Ok(flatten_results(results))
    }

    /// Write an entire buffer to a file synchronously.
    pub fn write_all(path: impl AsRef<Path>, data: Vec<u8>) -> IoResult<()> {
        let path_ref = path.as_ref();
        if data.is_empty() {
            let file = open_direct_write_sync(path_ref)?;
            file.sync_all()?;
            return Ok(());
        }

        let len = data.len();

        if can_use_direct_write(len) {
            let is_aligned = data.as_ptr().align_offset(BLOCK_SIZE) == 0;
            if is_aligned {
                let mut file = open_direct_write_sync(path_ref)?;
                file.write_all(&data)?;
                file.sync_all()
            } else {
                let mut buf = alloc_aligned(len)?;
                buf.extend_from_slice(&data);

                let mut file = open_direct_write_sync(path_ref)?;
                file.write_all(&buf)?;
                file.sync_all()
            }
        } else {
            let mut file = File::create(path_ref)?;
            file.write_all(&data)?;
            file.sync_all()
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod non_linux {
    use super::super::batch::{flatten_results, group_requests_by_file};
    use super::*;
    use std::fs::File;
    use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
    use std::thread;

    pub fn load(path: impl AsRef<Path>) -> IoResult<Vec<u8>> {
        let mut file = File::open(path.as_ref())?;
        let file_len = file.metadata()?.len();

        let len = usize::try_from(file_len)
            .map_err(|_foo| Error::new(ErrorKind::InvalidInput, "file too large"))?;

        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(buf.into_inner())
    }

    #[inline]
    pub fn load_parallel<P: AsRef<Path>>(path: P, chunks: usize) -> IoResult<Vec<u8>> {
        if chunks == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "chunks must be greater than 0",
            ));
        }

        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let metadata = file.metadata()?;
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
            let mut file_clone = file.try_clone()?;

            let handle = thread::spawn(move || {
                file_clone.seek(SeekFrom::Start(u64::try_from(start).map_err(|_e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek offset too large")
                })?))?;

                let slice = unsafe { buffer_slice.as_mut_slice() };
                file_clone.read_exact(slice)?;
                IoResult::Ok(())
            });

            handles.push(handle);
        }

        for handle in handles {
            handle
                .join()
                .map_err(|_e| std::io::Error::other("thread panicked"))??;
        }

        final_buf.truncate(file_size);
        Ok(final_buf.into_inner())
    }

    pub fn load_range(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
        let mut file = File::open(path.as_ref())?;
        file.seek(SeekFrom::Start(offset))?;

        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(buf.into_inner())
    }

    pub fn load_range_batch(
        requests: &[(impl AsRef<Path> + Send + Sync, u64, usize)],
    ) -> IoResult<Vec<super::super::batch::FlattenedResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);

        let mut results = Vec::with_capacity(requests.len());
        for (path, reqs) in grouped {
            let mut handles = Vec::with_capacity(reqs.len());

            for req in reqs {
                let path_clone = path.clone();
                handles.push(thread::spawn(move || {
                    let data = load_range(&path_clone, req.offset, req.len)?;
                    IoResult::Ok((req.idx, data, 0, req.len))
                }));
            }

            let mut file_results = Vec::with_capacity(handles.len());
            for handle in handles {
                file_results.push(
                    handle
                        .join()
                        .map_err(|_e| std::io::Error::other("thread panicked"))??,
                );
            }

            results.push(file_results);
        }

        Ok(flatten_results(results))
    }

    /// Write an entire buffer to a file synchronously.
    pub fn write_all(path: impl AsRef<Path>, data: Vec<u8>) -> IoResult<()> {
        if data.is_empty() {
            let mut file = File::create(path.as_ref())?;
            file.sync_all()?;
            return Ok(());
        }

        let mut file = File::create(path.as_ref())?;
        file.write_all(&data)?;
        file.sync_all()
    }
}

#[cfg(target_os = "linux")]
pub use linux::*;

#[cfg(not(target_os = "linux"))]
pub use non_linux::*;
