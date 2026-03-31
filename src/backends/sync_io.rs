//! Synchronous blocking I/O using std::fs.
//!
//! Platform-specific implementations:
//! - Linux: Uses O_DIRECT for aligned reads when possible, with chunked parallel reads for large files
//! - Other platforms: Simple buffered file reads

use super::{BatchRequest, IoResult};
use std::path::Path;

type IndexedLoadResult = (usize, std::sync::Arc<[u8]>, usize, usize);

// ---------------------------------------------------------------------------
// Linux implementation with O_DIRECT support
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod linux {
    use super::super::{get_buffer_pool, buffer_slice::BufferSlice, IoResult, batch::group_requests_by_file, byte::OwnedBytes};
    use super::IndexedLoadResult;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::thread;

    #[inline]
    const fn div_ceil(a: usize, b: usize) -> usize {
        a.div_ceil(b)
    }

    pub fn load(path: &Path) -> IoResult<OwnedBytes> {
        use super::super::odirect::{alloc_aligned as alloc_aligned_buf, can_use_direct_read, open_direct_read_sync};
        use super::super::{calculate_chunks, MAX_SINGLE_READ};

        let mut file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;

        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }

        if len > MAX_SINGLE_READ {
            return load_chunked(path, calculate_chunks(len));
        }

        if can_use_direct_read(len, len) {
            match open_direct_read_sync(path) {
                Ok(mut direct_file) => {
                    let mut buf = alloc_aligned_buf(len)?;
                    buf.set_len(len);
                    direct_file.read_exact(buf.as_mut_slice())?;
                    return Ok(OwnedBytes::Aligned(buf));
                }
                Err(err) if err.raw_os_error() == Some(libc::EINVAL) => {}
                Err(err) => return Err(err),
            }
        }

        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(OwnedBytes::Pooled(buf))
    }

    fn load_chunked(path: &Path, chunks: usize) -> IoResult<OwnedBytes> {
        use super::super::odirect::{alloc_aligned as alloc_aligned_buf, can_use_direct_read, open_direct_read_sync};

        if chunks == 0 {
            return Err(std::io::Error::other("chunks must be > 0"));
        }

        let file = File::open(path)?;
        let file_size = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        let chunk_size = div_ceil(file_size, chunks);

        if can_use_direct_read(file_size, chunk_size) {
            match open_direct_read_sync(path) {
                Ok(_) => {
                    drop(file);
                    let mut final_buf = alloc_aligned_buf(file_size)?;
                    final_buf.set_len(file_size);

                    let handles: Vec<_> = (0..chunks)
                        .map(|i| {
                            let start = i * chunk_size;
                            let end = std::cmp::min(start + chunk_size, file_size);
                            if start >= end {
                                return None;
                            }
                            let chunk_slice = final_buf.as_mut_slice().get_mut(start..end)?;
                            let mut buffer_slice = unsafe { BufferSlice::from_slice(chunk_slice) };
                            let path_clone = path.to_path_buf();
                            Some(thread::spawn(move || {
                                let mut f = open_direct_read_sync(path_clone.as_path())?;
                                f.seek(SeekFrom::Start(start as u64))?;
                                let s = unsafe { buffer_slice.as_mut_slice() };
                                f.read_exact(s)?;
                                IoResult::Ok(())
                            }))
                        })
                        .collect();

                    for h in handles.into_iter().flatten() {
                        h.join().map_err(|_| std::io::Error::other("thread panicked"))??;
                    }
                    return Ok(OwnedBytes::Aligned(final_buf));
                }
                Err(err) if err.raw_os_error() == Some(libc::EINVAL) => {}
                Err(err) => return Err(err),
            }
        }

        let mut final_buf = get_buffer_pool().get(file_size);
        let handles: Vec<_> = (0..chunks)
            .map(|i| {
                let start = i * chunk_size;
                let end = std::cmp::min(start + chunk_size, file_size);
                if start >= end {
                    return None;
                }
                let chunk_slice = final_buf.as_mut_slice().get_mut(start..end)?;
                let mut buffer_slice = unsafe { BufferSlice::from_slice(chunk_slice) };
                let path_clone = path.to_path_buf();
                Some(thread::spawn(move || {
                    let mut f = File::open(path_clone)?;
                    f.seek(SeekFrom::Start(start as u64))?;
                    let s = unsafe { buffer_slice.as_mut_slice() };
                    f.read_exact(s)?;
                    IoResult::Ok(())
                }))
            })
            .collect();

        for h in handles.into_iter().flatten() {
            h.join().map_err(|_| std::io::Error::other("thread panicked"))??;
        }
        final_buf.truncate(file_size);
        Ok(OwnedBytes::Pooled(final_buf))
    }

    pub fn load_range(path: &Path, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        use super::super::odirect::{alloc_aligned as alloc_aligned_buf, is_block_aligned, open_direct_read_sync};

        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }

        if is_block_aligned(offset, len) {
            match open_direct_read_sync(path) {
                Ok(mut file) => {
                    let mut buf = alloc_aligned_buf(len)?;
                    buf.set_len(len);
                    file.seek(SeekFrom::Start(offset))?;
                    file.read_exact(buf.as_mut_slice())?;
                    return Ok(OwnedBytes::Aligned(buf));
                }
                Err(err) if err.raw_os_error() == Some(libc::EINVAL) => {}
                Err(err) => return Err(err),
            }
        }

        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(OwnedBytes::Pooled(buf))
    }

    pub fn load_range_batch(requests: &[(PathBuf, u64, usize)]) -> IoResult<Vec<IndexedLoadResult>> {
        use rayon::prelude::*;

        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);

        let results = grouped
            .into_par_iter()
            .map(|(path, reqs)| {
                reqs.par_iter()
                    .map(|req| {
                        let path = path.clone();
                        let data = load_range(&path, req.offset, req.len)?;
                        Ok((req.idx, data.into_shared(), 0, req.len))
                    })
                    .collect::<Result<Vec<_>, std::io::Error>>()
            })
            .collect::<Result<Vec<Vec<IndexedLoadResult>>, std::io::Error>>();

        let mut indexed: Vec<IndexedLoadResult> = results?.into_iter().flatten().collect();
        indexed.sort_by_key(|(idx, _, _, _)| *idx);
        Ok(indexed)
    }
}

// ---------------------------------------------------------------------------
// Portable implementation (non-Linux)
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "linux"))]
mod portable {
    use super::super::get_buffer_pool;
    use super::super::batch::group_requests_by_file;
    use super::super::byte::OwnedBytes;
    use super::IndexedLoadResult;
    use super::IoResult;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    pub fn load(path: &Path) -> IoResult<OwnedBytes> {
        let mut file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(OwnedBytes::Pooled(buf))
    }

    pub fn load_range(path: &Path, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(OwnedBytes::Pooled(buf))
    }

    pub fn load_range_batch(requests: &[(PathBuf, u64, usize)]) -> IoResult<Vec<IndexedLoadResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);
        let mut indexed: Vec<IndexedLoadResult> = Vec::with_capacity(requests.len());

        for (path, reqs) in grouped {
            for req in reqs {
                let data = load_range(&path, req.offset, req.len)?;
                indexed.push((req.idx, data.into_shared(), 0, req.len));
            }
        }

        indexed.sort_by_key(|(idx, _, _, _)| *idx);
        Ok(indexed)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub(crate) struct SyncReaderEngine;

impl SyncReaderEngine {
    pub(crate) const fn new() -> Self {
        Self
    }

    pub(crate) fn load(&mut self, path: impl AsRef<Path>) -> IoResult<super::byte::OwnedBytes> {
        #[cfg(target_os = "linux")]
        {
            linux::load(path.as_ref())
        }
        #[cfg(not(target_os = "linux"))]
        {
            portable::load(path.as_ref())
        }
    }

    pub(crate) fn load_batch(
        &mut self,
        paths: &[std::path::PathBuf],
    ) -> IoResult<Vec<super::byte::OwnedBytes>> {
        #[cfg(target_os = "linux")]
        {
            use rayon::prelude::*;

            paths.par_iter()
                .map(|path| linux::load(path))
                .collect()
        }
        #[cfg(not(target_os = "linux"))]
        {
            paths.iter().map(|path| portable::load(path)).collect()
        }
    }

    pub(crate) fn load_range(&mut self, path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<super::byte::OwnedBytes> {
        #[cfg(target_os = "linux")]
        {
            linux::load_range(path.as_ref(), offset, len)
        }
        #[cfg(not(target_os = "linux"))]
        {
            portable::load_range(path.as_ref(), offset, len)
        }
    }

    pub(crate) fn load_range_batch(&mut self, requests: &[BatchRequest]) -> IoResult<Vec<super::batch::FlattenedResult>> {
        #[cfg(target_os = "linux")]
        {
            let indexed = linux::load_range_batch(requests)?;
            Ok(indexed.into_iter().map(|(_, data, offset, len)| (data, offset, len)).collect())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let indexed = portable::load_range_batch(requests)?;
            Ok(indexed.into_iter().map(|(_, data, offset, len)| (data, offset, len)).collect())
        }
    }
}

pub(crate) struct SyncWriterEngine {
    file: std::fs::File,
}

impl std::fmt::Debug for SyncWriterEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncWriterEngine").finish_non_exhaustive()
    }
}

impl SyncWriterEngine {
    pub(crate) fn create(path: &Path) -> IoResult<Self> {
        if let Some(parent) = path.parent() && !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        Ok(Self { file })
    }

    pub(crate) fn write_all(&mut self, data: &[u8]) -> IoResult<()> {
        use std::io::{Seek, SeekFrom, Write};
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(data)
    }

    pub(crate) fn write_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()> {
        use std::io::{Seek, SeekFrom, Write};
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(data)
    }

    pub(crate) fn sync_all(&mut self) -> IoResult<()> {
        self.file.sync_all()
    }
}
