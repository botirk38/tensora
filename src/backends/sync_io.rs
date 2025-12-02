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

    use super::super::odirect::{
        BLOCK_SIZE, alloc_aligned, can_use_direct_read, can_use_direct_write, is_block_aligned,
        open_direct_read_sync, open_direct_write_sync,
    };
    use super::*;
    use rayon::prelude::*;
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
        Ok(buf.to_vec())
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
                        let path_clone = path_ref.to_path_buf();

                        let handle = thread::spawn(move || {
                            let mut direct_file = open_direct_read_sync(path_clone.as_path())?;
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
        Ok(final_buf.to_vec())
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
        Ok(buf.to_vec())
    }

    pub fn load_range_batch(
        requests: &[(impl AsRef<Path> + Send + Sync, u64, usize)],
    ) -> IoResult<Vec<super::super::batch::FlattenedResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        let mut results: Vec<(usize, Vec<u8>, usize, usize)> = pool.install(|| {
            requests
                .par_iter()
                .enumerate()
                .map(|(idx, (path, offset, len))| {
                    load_range(path, *offset, *len).map(|data| (idx, data, 0, *len))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        results.sort_by_key(|(idx, _, _, _)| *idx);

        Ok(results
            .into_iter()
            .map(|(_, data, _, len)| (data, 0, len))
            .collect())
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

    use super::*;
    use rayon::prelude::*;
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
        Ok(buf.to_vec())
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
        Ok(buf.to_vec())
    }

    pub fn load_range_batch(
        requests: &[(impl AsRef<Path> + Send + Sync, u64, usize)],
    ) -> IoResult<Vec<super::super::batch::FlattenedResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        let mut results: Vec<(usize, Vec<u8>, usize, usize)> = pool.install(|| {
            requests
                .par_iter()
                .enumerate()
                .map(|(idx, (path, offset, len))| {
                    load_range(path, *offset, *len).map(|data| (idx, data, 0, *len))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        results.sort_by_key(|(idx, _, _, _)| *idx);

        Ok(results
            .into_iter()
            .map(|(_, data, _, len)| (data, 0, len))
            .collect())
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
    }

    // -----------------------------------------------------------------------
    // Integration Tests - Cross-Platform
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_empty_file() {
        let tmpfile = NamedTempFile::new().unwrap();
        let result = load(tmpfile.path()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_load_small_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"test data").unwrap();
        tmpfile.flush().unwrap();

        let result = load(tmpfile.path()).unwrap();
        assert_eq!(result, b"test data");
    }

    #[test]
    fn test_load_larger_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = vec![0xAB; 1024 * 1024]; // 1MB
        tmpfile.write_all(&data).unwrap();
        tmpfile.flush().unwrap();

        let result = load(tmpfile.path()).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_load_parallel_basic() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = vec![0xCD; 1024 * 100]; // 100KB
        tmpfile.write_all(&data).unwrap();
        tmpfile.as_file().sync_all().unwrap(); // Ensure data is written
        tmpfile.flush().unwrap();

        let result = load_parallel(tmpfile.path(), 4).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_load_parallel_single_chunk() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = vec![0xEF; 1024 * 5];
        tmpfile.write_all(&data).unwrap();
        tmpfile.flush().unwrap();

        let result = load_parallel(tmpfile.path(), 1).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_load_parallel_zero_chunks_error() {
        let tmpfile = NamedTempFile::new().unwrap();
        let result = load_parallel(tmpfile.path(), 0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_load_parallel_empty_file() {
        let tmpfile = NamedTempFile::new().unwrap();
        let result = load_parallel(tmpfile.path(), 4).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_load_parallel_vs_sequential() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = vec![0x42; 1024 * 500]; // 500KB
        tmpfile.write_all(&data).unwrap();
        tmpfile.as_file().sync_all().unwrap(); // Ensure data is written
        tmpfile.flush().unwrap();

        let sequential = load(tmpfile.path()).unwrap();
        let parallel = load_parallel(tmpfile.path(), 8).unwrap();

        assert_eq!(sequential, parallel);
        assert_eq!(sequential, data);
    }

    #[test]
    fn test_load_range_basic() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        tmpfile.write_all(data).unwrap();
        tmpfile.flush().unwrap();

        let result = load_range(tmpfile.path(), 5, 10).unwrap();
        assert_eq!(result, &data[5..15]);
    }

    #[test]
    fn test_load_range_full_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = b"complete file content";
        tmpfile.write_all(data).unwrap();
        tmpfile.flush().unwrap();

        let result = load_range(tmpfile.path(), 0, data.len()).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_load_range_zero_length() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(b"test").unwrap();
        tmpfile.flush().unwrap();

        let result = load_range(tmpfile.path(), 0, 0).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_write_all_basic() {
        let tmpfile = NamedTempFile::new().unwrap();
        let data = b"Hello, sync_io!".to_vec();

        write_all(tmpfile.path(), data.clone()).unwrap();

        // Read back and verify
        let result = load(tmpfile.path()).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_write_all_empty() {
        let tmpfile = NamedTempFile::new().unwrap();
        write_all(tmpfile.path(), Vec::new()).unwrap();

        let result = load(tmpfile.path()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_write_all_large() {
        let tmpfile = NamedTempFile::new().unwrap();
        let data = vec![0x55; 1024 * 1024]; // 1MB

        write_all(tmpfile.path(), data.clone()).unwrap();

        let result = load(tmpfile.path()).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_load_range_batch_single_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        let data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        tmpfile.write_all(data).unwrap();
        tmpfile.flush().unwrap();

        let requests = vec![
            (tmpfile.path(), 0, 10),
            (tmpfile.path(), 10, 10),
            (tmpfile.path(), 20, 10),
        ];

        let results = load_range_batch(&requests).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, &data[0..10]);
        assert_eq!(results[1].0, &data[10..20]);
        assert_eq!(results[2].0, &data[20..30]);
    }

    #[test]
    fn test_load_range_batch_multiple_files() {
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

        let results = load_range_batch(&requests).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, b"FILE1");
        assert_eq!(results[1].0, b"FILE2");
        assert_eq!(results[2].0, b"DATA");
    }

    #[test]
    fn test_load_range_batch_empty() {
        let requests: Vec<(&str, u64, usize)> = vec![];
        let results = load_range_batch(&requests).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_load_missing_file() {
        let result = load("/nonexistent/file/path");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Linux-Specific Tests
    // -----------------------------------------------------------------------

    #[cfg(target_os = "linux")]
    mod linux_tests {
        use super::*;

        #[test]
        fn test_load_with_direct_io_attempt() {
            // This test verifies that O_DIRECT path is attempted
            // It may fall back to buffered I/O depending on filesystem
            let mut tmpfile = NamedTempFile::new().unwrap();
            let data = vec![0x77; 4096 * 2]; // Block-aligned size
            tmpfile.write_all(&data).unwrap();
            tmpfile.flush().unwrap();

            let result = load(tmpfile.path()).unwrap();
            assert_eq!(result, data);
        }

        #[test]
        fn test_load_parallel_direct_io() {
            let mut tmpfile = NamedTempFile::new().unwrap();
            let data = vec![0x88; 4096 * 100]; // Block-aligned
            tmpfile.write_all(&data).unwrap();
            tmpfile.as_file().sync_all().unwrap(); // Ensure data is written
            tmpfile.flush().unwrap();

            let result = load_parallel(tmpfile.path(), 4).unwrap();
            assert_eq!(result, data);
        }

        #[test]
        fn test_write_all_aligned_data() {
            let tmpfile = NamedTempFile::new().unwrap();
            let data = vec![0x99; 4096]; // Block-aligned

            write_all(tmpfile.path(), data.clone()).unwrap();

            let result = load(tmpfile.path()).unwrap();
            assert_eq!(result, data);
        }
    }
}
