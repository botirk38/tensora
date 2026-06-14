//! Synchronous blocking storage engine.
//!
//! [`SyncStorage`] implements [`ReadableStorage`] and [`StorageEngine`].
//! On Linux it defaults to O_DIRECT where possible; on other platforms it
//! uses buffered `std::fs` I/O.
//!
//! Write access is obtained by calling [`SyncStorage::create_writer`], which
//! returns a [`SyncWriter`] that holds an open file handle and implements
//! [`WritableStorage`].

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use crate::storage::{
    FileReadRequest, IoResult, RangeReadRequest, WriteAtRequest,
    availability::{StorageAvailability, StorageCapabilities, StorageKind},
    buffer::OwnedBytes,
};

type IndexedRangeRead = (usize, Arc<[u8]>, usize, usize);

// ============================================================================
// SyncStorage
// ============================================================================

/// Synchronous blocking storage engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyncStorage;

impl SyncStorage {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub fn create_writer(&self, path: &Path) -> IoResult<SyncWriter> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        Ok(SyncWriter { file })
    }
}

impl super::StorageEngine for SyncStorage {
    fn kind(&self) -> StorageKind {
        StorageKind::Sync
    }

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        StorageAvailability::Available
    }

    fn capabilities() -> StorageCapabilities
    where
        Self: Sized,
    {
        StorageCapabilities::probe()
    }
}

impl super::ReadableStorage for SyncStorage {
    fn read_file(&self, req: FileReadRequest<'_>) -> IoResult<OwnedBytes> {
        load(req.path)
    }

    fn read_range(&self, req: RangeReadRequest<'_>) -> IoResult<OwnedBytes> {
        load_range(req.path, req.offset, req.len)
    }

    fn read_ranges(
        &self,
        req: crate::storage::BatchReadRequest<'_>,
    ) -> IoResult<Vec<crate::storage::RangeReadResult>> {
        if req.is_empty() {
            return Ok(Vec::new());
        }
        let requests: Vec<(std::path::PathBuf, u64, usize)> = req
            .paths
            .iter()
            .zip(req.ranges.iter())
            .map(|(p, r)| (p.to_path_buf(), r.offset, r.len))
            .collect();
        let mut indexed = load_range_batch_indexed(&requests)?;
        indexed.sort_by_key(|(i, _, _, _)| *i);
        Ok(indexed
            .into_iter()
            .map(|(i, bytes, lo, ll)| crate::storage::RangeReadResult {
                request_index: i,
                bytes,
                logical_offset: lo,
                logical_len: ll,
            })
            .collect())
    }
}

// ============================================================================
// SyncWriter + WritableStorage
// ============================================================================

pub struct SyncWriter {
    file: std::fs::File,
}

impl std::fmt::Debug for SyncWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncWriter").finish_non_exhaustive()
    }
}

impl super::StorageEngine for SyncWriter {
    fn kind(&self) -> StorageKind {
        StorageKind::Sync
    }

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        StorageAvailability::Available
    }

    fn capabilities() -> StorageCapabilities
    where
        Self: Sized,
    {
        StorageCapabilities::probe()
    }
}

impl super::WritableStorage for SyncWriter {
    fn write_at(&mut self, req: WriteAtRequest<'_>) -> IoResult<()> {
        self.file.seek(SeekFrom::Start(req.offset))?;
        self.file.write_all(req.data)
    }

    fn flush(&mut self) -> IoResult<()> {
        self.file.sync_all()
    }
}

// ============================================================================
// Core I/O functions
// ============================================================================

fn load(path: &Path) -> IoResult<OwnedBytes> {
    #[cfg(target_os = "linux")]
    return linux::load(path);
    #[cfg(not(target_os = "linux"))]
    return portable::load(path);
}

fn load_range(path: &Path, offset: u64, len: usize) -> IoResult<OwnedBytes> {
    if len == 0 {
        return Ok(OwnedBytes::Shared(Arc::new([])));
    }
    #[cfg(target_os = "linux")]
    return linux::load_range(path, offset, len);
    #[cfg(not(target_os = "linux"))]
    return portable::load_range(path, offset, len);
}

fn load_range_batch_indexed(
    requests: &[(std::path::PathBuf, u64, usize)],
) -> IoResult<Vec<IndexedRangeRead>> {
    #[cfg(target_os = "linux")]
    return linux::load_range_batch(requests);
    #[cfg(not(target_os = "linux"))]
    return portable::load_range_batch(requests);
}

// ============================================================================
// Linux: O_DIRECT-aware implementation
// ============================================================================

#[cfg(target_os = "linux")]
mod linux {
    use super::super::buffer::{AlignedBuffer, OwnedBytes, get_buffer_pool};
    use super::{Arc, IndexedRangeRead};
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::thread;

    const BLOCK_SIZE: usize = 4096;
    const BLOCK_SIZE_U64: u64 = 4096;
    const MAX_SINGLE_READ: usize = 512 * 1024 * 1024;

    fn round_up_to_block(n: usize) -> usize {
        (n + BLOCK_SIZE - 1) & !(BLOCK_SIZE - 1)
    }

    fn open_prefer_direct(path: &Path) -> std::io::Result<(std::fs::File, bool)> {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
        {
            Ok(f) => Ok((f, true)),
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                Ok((std::fs::File::open(path)?, false))
            }
            Err(e) => Err(e),
        }
    }

    fn read_direct(
        file: &mut std::fs::File,
        buf: &mut [u8],
        actual_len: usize,
    ) -> std::io::Result<()> {
        let mut pos = 0;
        while pos < actual_len {
            let n = file.read(&mut buf[pos..])?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("O_DIRECT short read: got {pos} of {actual_len} bytes"),
                ));
            }
            pos += n;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<OwnedBytes> {
        let (mut file, direct) = open_prefer_direct(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        if len > MAX_SINGLE_READ {
            let chunks = (len / (128 * 1024 * 1024)).clamp(1, 8);
            return load_chunked(path, chunks);
        }
        if direct {
            let aligned_len = round_up_to_block(len);
            let mut buf = AlignedBuffer::new(aligned_len)?;
            buf.set_len(aligned_len);
            read_direct(&mut file, buf.as_mut_slice(), len)?;
            buf.set_len(len);
            Ok(OwnedBytes::Aligned(buf))
        } else {
            let mut buf = get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(OwnedBytes::Pooled(buf))
        }
    }

    fn load_chunked(path: &Path, chunks: usize) -> std::io::Result<OwnedBytes> {
        let (file, direct) = open_prefer_direct(path)?;
        let file_size = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        let raw_chunk = file_size.div_ceil(chunks).max(1);
        let chunk_size = if direct {
            round_up_to_block(raw_chunk)
        } else {
            raw_chunk
        };
        drop(file);

        if direct {
            let aligned_total = round_up_to_block(file_size);
            let mut final_buf = AlignedBuffer::new(aligned_total)?;
            final_buf.set_len(aligned_total);
            struct SendSlice(*mut u8, usize);
            unsafe impl Send for SendSlice {}
            let handles: Vec<_> = (0..chunks)
                .filter_map(|i| {
                    let start = i * chunk_size;
                    let end = std::cmp::min(start + chunk_size, file_size);
                    if start >= end {
                        return None;
                    }
                    let read_len = round_up_to_block(end - start);
                    let buf_end = std::cmp::min(start + read_len, aligned_total);
                    let slice = final_buf.as_mut_slice().get_mut(start..buf_end)?;
                    let ptr = SendSlice(slice.as_mut_ptr(), slice.len());
                    let actual_len = end - start;
                    let path_clone = path.to_path_buf();
                    Some(thread::spawn(move || {
                        let buf = unsafe { std::slice::from_raw_parts_mut(ptr.0, ptr.1) };
                        let (mut f, _) = open_prefer_direct(&path_clone)?;
                        f.seek(SeekFrom::Start(start as u64))?;
                        read_direct(&mut f, buf, actual_len)?;
                        std::io::Result::Ok(())
                    }))
                })
                .collect();
            for h in handles {
                h.join()
                    .map_err(|_| std::io::Error::other("thread panicked"))??
            }
            final_buf.set_len(file_size);
            Ok(OwnedBytes::Aligned(final_buf))
        } else {
            let mut final_buf = get_buffer_pool().get(file_size);
            struct SendSlice(*mut u8, usize);
            unsafe impl Send for SendSlice {}
            let handles: Vec<_> = (0..chunks)
                .filter_map(|i| {
                    let start = i * chunk_size;
                    let end = std::cmp::min(start + chunk_size, file_size);
                    if start >= end {
                        return None;
                    }
                    let slice = final_buf.as_mut_slice().get_mut(start..end)?;
                    let ptr = SendSlice(slice.as_mut_ptr(), slice.len());
                    let path_clone = path.to_path_buf();
                    Some(thread::spawn(move || {
                        let buf = unsafe { std::slice::from_raw_parts_mut(ptr.0, ptr.1) };
                        let mut f = std::fs::File::open(&path_clone)?;
                        f.seek(SeekFrom::Start(start as u64))?;
                        f.read_exact(buf)?;
                        std::io::Result::Ok(())
                    }))
                })
                .collect();
            for h in handles {
                h.join()
                    .map_err(|_| std::io::Error::other("thread panicked"))??
            }
            final_buf.truncate(file_size);
            Ok(OwnedBytes::Pooled(final_buf))
        }
    }

    pub fn load_range(path: &Path, offset: u64, len: usize) -> std::io::Result<OwnedBytes> {
        let (mut file, direct) = open_prefer_direct(path)?;
        if direct {
            let aligned_offset = offset & !(BLOCK_SIZE_U64 - 1);
            let head_skip = (offset - aligned_offset) as usize;
            let aligned_len = round_up_to_block(head_skip + len);
            file.seek(SeekFrom::Start(aligned_offset))?;
            let mut buf = AlignedBuffer::new(aligned_len)?;
            buf.set_len(aligned_len);
            read_direct(&mut file, buf.as_mut_slice(), head_skip + len)?;
            if head_skip == 0 {
                buf.set_len(len);
                return Ok(OwnedBytes::Aligned(buf));
            }
            let slice = &buf.as_slice()[head_skip..head_skip + len];
            Ok(OwnedBytes::Shared(Arc::from(slice)))
        } else {
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(OwnedBytes::Pooled(buf))
        }
    }

    pub fn load_range_batch(
        requests: &[(PathBuf, u64, usize)],
    ) -> std::io::Result<Vec<IndexedRangeRead>> {
        use rayon::prelude::*;
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        requests
            .par_iter()
            .enumerate()
            .map(|(i, (path, offset, len))| {
                let data = if *len == 0 {
                    Arc::<[u8]>::from(vec![])
                } else {
                    load_range(path, *offset, *len)?.into_shared()
                };
                Ok((i, data, 0, *len))
            })
            .collect()
    }
}

// ============================================================================
// Portable: plain buffered std::fs I/O (non-Linux)
// ============================================================================

#[cfg(not(target_os = "linux"))]
mod portable {
    use super::super::buffer::{OwnedBytes, get_buffer_pool};
    use super::{Arc, IndexedRangeRead};
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};

    pub fn load(path: &Path) -> std::io::Result<OwnedBytes> {
        let mut file = std::fs::File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(OwnedBytes::Pooled(buf))
    }

    pub fn load_range(path: &Path, offset: u64, len: usize) -> std::io::Result<OwnedBytes> {
        let mut file = std::fs::File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(OwnedBytes::Pooled(buf))
    }

    pub fn load_range_batch(
        requests: &[(PathBuf, u64, usize)],
    ) -> std::io::Result<Vec<IndexedRangeRead>> {
        requests
            .iter()
            .enumerate()
            .map(|(i, (path, offset, len))| {
                let data = if *len == 0 {
                    Arc::<[u8]>::from(vec![])
                } else {
                    load_range(path, *offset, *len)?.into_shared()
                };
                Ok((i, data, 0, *len))
            })
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        BatchRange, BatchReadRequest, ReadableStorage, StorageEngine, WritableStorage,
    };
    use tempfile::TempDir;

    fn write_tmp(dir: &TempDir, name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn read_file_roundtrip() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let path = write_tmp(&dir, "file.bin", &data);

        let result = SyncStorage::default()
            .read_file(FileReadRequest::new(&path))
            .unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[test]
    fn read_file_empty() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "empty.bin", b"");

        let result = SyncStorage::default()
            .read_file(FileReadRequest::new(&path))
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_range_returns_correct_slice() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..100).collect();
        let path = write_tmp(&dir, "range.bin", &data);

        let result = SyncStorage::default()
            .read_range(RangeReadRequest::new(&path, 10, 20))
            .unwrap();
        assert_eq!(result.as_ref(), &data[10..30]);
    }

    #[test]
    fn read_range_zero_len() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "z.bin", b"hello");

        let result = SyncStorage::default()
            .read_range(RangeReadRequest::new(&path, 0, 0))
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_ranges_empty() {
        let results = SyncStorage::default()
            .read_ranges(BatchReadRequest::new(&[], &[]))
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn read_ranges_single() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..200).collect();
        let path = write_tmp(&dir, "batch.bin", &data);

        let paths = [path.as_path()];
        let ranges = [BatchRange::new(50, 30)];
        let results = SyncStorage::default()
            .read_ranges(BatchReadRequest::new(&paths, &ranges))
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data(), &data[50..80]);
    }

    #[test]
    fn read_ranges_multiple_preserves_order() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        let path = write_tmp(&dir, "multi.bin", &data);
        let p = path.as_path();

        let paths = [p, p, p];
        let ranges = [
            BatchRange::new(0, 10),
            BatchRange::new(20, 10),
            BatchRange::new(100, 5),
        ];
        let results = SyncStorage::default()
            .read_ranges(BatchReadRequest::new(&paths, &ranges))
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data(), &data[0..10]);
        assert_eq!(results[1].data(), &data[20..30]);
        assert_eq!(results[2].data(), &data[100..105]);
    }

    #[test]
    fn kind_is_sync() {
        assert_eq!(SyncStorage::new().kind(), StorageKind::Sync);
    }

    #[test]
    fn availability_is_available() {
        assert!(SyncStorage::availability().is_available());
    }

    #[test]
    fn write_at_and_flush_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let mut writer = SyncStorage::new().create_writer(&path).unwrap();
        writer
            .write_at(WriteAtRequest::new(0, b"hello world"))
            .unwrap();
        writer.flush().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello world");
    }
}
