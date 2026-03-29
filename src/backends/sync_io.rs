//! Synchronous blocking I/O using std::fs

use super::{buffer_slice::BufferSlice, get_buffer_pool, IoResult, SyncBackend};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

static RAYON_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

type IndexedLoadResult = (usize, Arc<[u8]>, usize, usize);

fn get_rayon_pool() -> &'static rayon::ThreadPool {
    RAYON_POOL.get_or_init(|| {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap()
    })
}

/// Ceiling division: (a + b - 1) / b
#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Default synchronous backend using std::fs.
#[derive(Clone, Copy, Debug)]
pub struct DefaultSyncBackend;

impl SyncBackend for DefaultSyncBackend {
    fn load(&self, path: &Path) -> IoResult<Vec<u8>> {
        load(path)
    }

    fn load_parallel(&self, path: &Path, chunks: usize) -> IoResult<Vec<u8>> {
        load_parallel(path, chunks)
    }

    fn load_range(&self, path: &Path, offset: u64, len: usize) -> IoResult<Vec<u8>> {
        load_range(path, offset, len)
    }

    fn load_range_batch(
        &self,
        requests: &[super::BatchRequest],
    ) -> IoResult<Vec<super::batch::FlattenedResult>> {
        let owned: Vec<(PathBuf, u64, usize)> = requests
            .iter()
            .map(|(path, offset, len)| (path.clone(), *offset, *len))
            .collect();
        load_range_batch(&owned)
    }

    fn write_all(&self, path: &Path, data: Vec<u8>) -> IoResult<()> {
        write_all(path, data)
    }
}

#[cfg(target_os = "linux")]
mod linux {

    use super::super::odirect::{
        alloc_aligned, can_use_direct_read, can_use_direct_write, is_block_aligned,
        open_direct_read_sync, open_direct_write_sync, BLOCK_SIZE,
    };
    use super::*;
    use rayon::prelude::*;
    use std::fs::File;
    use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
    use std::thread;

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

    /// Loads a file using parallel threads for improved throughput.
    ///
    /// # Errors
    ///
    /// - `chunks` is zero
    /// - File cannot be opened or read
    /// - File size exceeds `usize` limits
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
            // Open a fresh file handle for each thread to avoid shared seek position
            let path_clone = path_ref.to_path_buf();

            let handle = thread::spawn(move || {
                let mut thread_file = File::open(&path_clone)?;
                thread_file.seek(SeekFrom::Start(u64::try_from(start).map_err(|_e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek offset too large")
                })?))?;

                let slice = unsafe { buffer_slice.as_mut_slice() };
                thread_file.read_exact(slice)?;
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

    /// Loads a range of bytes from a file at the specified offset.
    ///
    /// # Errors
    ///
    /// - File cannot be opened or read
    /// - Seek or read operation fails
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

        let pool = get_rayon_pool();
        let results: Vec<(usize, Vec<u8>, usize, usize)> = pool.install(|| {
            requests
                .par_iter()
                .enumerate()
                .map(|(idx, (path, offset, len))| {
                    load_range(path, *offset, *len).map(|data| (idx, data, 0, *len))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        // Preallocate output in original order
        let mut output = Vec::with_capacity(results.len());
        let mut sorted = results;
        sorted.sort_by_key(|(idx, _, _, _)| *idx);
        for (_, data, _, len) in sorted {
            output.push((data, 0, len));
        }

        Ok(output)
    }

    /// Write an entire buffer to a file synchronously using Direct I/O when possible.
    ///
    /// # Errors
    ///
    /// - File cannot be created or written to
    /// - Sync operation fails
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
            // Open a fresh file handle for each thread to avoid shared seek position
            let path_clone = path_ref.to_path_buf();

            let handle = thread::spawn(move || {
                let mut thread_file = File::open(&path_clone)?;
                thread_file.seek(SeekFrom::Start(u64::try_from(start).map_err(|_e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek offset too large")
                })?))?;

                let slice = unsafe { buffer_slice.as_mut_slice() };
                thread_file.read_exact(slice)?;
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

    fn load_range_from_file(
        file: &mut File,
        requests: &[super::super::batch::IndexedRequest],
    ) -> IoResult<Vec<IndexedLoadResult>> {
        let mut results = Vec::with_capacity(requests.len());
        let mut sorted_reqs = requests.to_vec();
        sorted_reqs.sort_by_key(|r| r.offset);

        for req in sorted_reqs {
            file.seek(SeekFrom::Start(req.offset))?;
            let mut buf = get_buffer_pool().get(req.len);
            file.read_exact(&mut buf[..])?;
            let owned: Arc<[u8]> = buf.into_inner().into();
            results.push((req.idx, owned, 0, req.len));
        }

        results.sort_by_key(|(idx, _, _, _)| *idx);
        Ok(results)
    }

    pub fn load_range_batch(
        requests: &[(impl AsRef<Path> + Send + Sync, u64, usize)],
    ) -> IoResult<Vec<super::super::batch::FlattenedResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = super::super::batch::group_requests_by_file(requests);

        let pool = get_rayon_pool();
        let file_results: Vec<Vec<IndexedLoadResult>> = pool.install(|| {
            grouped
                .par_iter()
                .map(|(path, indexed_reqs)| {
                    let mut file = File::open(path)?;
                    load_range_from_file(&mut file, indexed_reqs)
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        let mut indexed: Vec<IndexedLoadResult> = file_results.into_iter().flatten().collect();
        indexed.sort_by_key(|(idx, _, _, _)| *idx);

        Ok(indexed
            .into_iter()
            .map(|(_, data, _, len)| (data, 0, len))
            .collect())
    }

    /// Write an entire buffer to a file synchronously.
    pub fn write_all(path: impl AsRef<Path>, data: Vec<u8>) -> IoResult<()> {
        if data.is_empty() {
            let file = File::create(path.as_ref())?;
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
        assert_eq!(results[0].0.as_ref(), &data[0..10]);
        assert_eq!(results[1].0.as_ref(), &data[10..20]);
        assert_eq!(results[2].0.as_ref(), &data[20..30]);
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
        assert_eq!(results[0].0.as_ref(), b"FILE1");
        assert_eq!(results[1].0.as_ref(), b"FILE2");
        assert_eq!(results[2].0.as_ref(), b"DATA");
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
    // Property-Based Tests
    // -----------------------------------------------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_div_ceil_properties(a in 1usize..10000, b in 1usize..100) {
                let result = div_ceil(a, b);
                // Property 1: result * b >= a (covers the range)
                prop_assert!(result * b >= a);
                // Property 2: (result - 1) * b < a (is minimal)
                if result > 0 {
                    prop_assert!((result - 1) * b < a);
                }
            }

            #[test]
            fn test_load_parallel_consistency(
                data_size in 128usize..4096,
                chunk_count in 1usize..8
            ) {
                // Skip edge cases where chunk_count exceeds data_size (leads to underflow)
                if chunk_count > data_size {
                    return Ok(());
                }

                let mut tmpfile = NamedTempFile::new().unwrap();
                let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
                tmpfile.write_all(&data).unwrap();
                tmpfile.as_file().sync_all().unwrap();
                tmpfile.flush().unwrap();

                // Sequential load
                let sequential = load(tmpfile.path()).unwrap();

                // Parallel load
                let parallel = load_parallel(tmpfile.path(), chunk_count).unwrap();

                // Property: Results should be identical
                prop_assert_eq!(&sequential, &parallel);
                prop_assert_eq!(&parallel, &data);
            }

            #[test]
            fn test_load_range_subslice_property(
                total_size in 10usize..1000,
                offset in 0usize..500,
                len in 1usize..200
            ) {
                // Ensure offset + len doesn't exceed total_size
                let offset = offset % (total_size - 1);
                let len = len.min(total_size - offset);

                let mut tmpfile = NamedTempFile::new().unwrap();
                let data: Vec<u8> = (0..total_size).map(|i| (i % 256) as u8).collect();
                tmpfile.write_all(&data).unwrap();
                tmpfile.flush().unwrap();

                let range_result = load_range(tmpfile.path(), offset as u64, len).unwrap();

                // Property: load_range should equal the corresponding slice
                prop_assert_eq!(range_result, &data[offset..offset + len]);
            }

            #[test]
            fn test_write_all_roundtrip(data_size in 1usize..10000) {
                let tmpfile = NamedTempFile::new().unwrap();
                let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();

                write_all(tmpfile.path(), data.clone()).unwrap();
                let loaded = load(tmpfile.path()).unwrap();

                // Property: Write then read should preserve data exactly
                prop_assert_eq!(loaded, data);
            }
        }
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
