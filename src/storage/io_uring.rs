//! Linux io_uring storage implementation with batched I/O.

use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;

use io_uring::{IoUring, opcode, types};

use crate::storage::{
    ByteRange, FileRange, IoResult, RangeRead, StorageEngine,
    availability::{StorageAvailability, StorageKind},
    buffer::OwnedBytes,
};

const RING_DEPTH: u32 = 256;
const MAX_IO_LEN: usize = u32::MAX as usize;

// ============================================================================
// IoUringStorage
// ============================================================================

/// High-throughput io_uring storage engine (Linux only).
#[derive(Debug, Clone, Copy, Default)]
pub struct IoUringStorage;

impl IoUringStorage {
    /// Create a new `IoUringStorage` engine.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn ensure_available() -> IoResult<()> {
        match IoUringStorage::availability() {
            StorageAvailability::Available => Ok(()),
            unavailable => Err(Error::other(format!("io_uring storage is {unavailable}"))),
        }
    }

    fn read_exact_at(file: &File, offset: u64, mut buf: &mut [u8]) -> IoResult<()> {
        let mut ring = IoUring::new(RING_DEPTH)?;
        let mut absolute_offset = offset;

        while !buf.is_empty() {
            let chunk_len = buf.len().min(MAX_IO_LEN);
            let (chunk, rest) = buf.split_at_mut(chunk_len);
            let mut read = 0usize;

            while read < chunk.len() {
                let len = (chunk.len() - read).min(MAX_IO_LEN);
                // SAFETY: `read < chunk.len()` and `len` is bounded by the
                // remaining bytes, so the resulting pointer stays within
                // `chunk` for the submitted read.
                let ptr = unsafe { chunk.as_mut_ptr().add(read) };
                let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), ptr, len as u32)
                    .offset(absolute_offset + read as u64)
                    .build()
                    .user_data(0);

                Self::submit_one(&mut ring, &entry)?;
                let result = Self::wait_one(&mut ring)?;
                if result == 0 {
                    return Err(Error::new(ErrorKind::UnexpectedEof, "short io_uring read"));
                }
                read += result;
            }

            absolute_offset = absolute_offset
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "read offset overflow"))?;
            buf = rest;
        }

        Ok(())
    }

    fn submit_one(ring: &mut IoUring, entry: &io_uring::squeue::Entry) -> IoResult<()> {
        {
            let mut submission = ring.submission();
            // SAFETY: `entry` references buffers owned by the caller and those
            // buffers remain alive until `submit_and_wait` completes below.
            unsafe {
                submission
                    .push(entry)
                    .map_err(|_| Error::other("io_uring submission queue is full"))?;
            }
        }
        ring.submit_and_wait(1)?;
        Ok(())
    }

    fn wait_one(ring: &mut IoUring) -> IoResult<usize> {
        let completion = ring
            .completion()
            .next()
            .ok_or_else(|| Error::other("io_uring completion queue was empty"))?;
        let result = completion.result();
        if result < 0 {
            return Err(Error::from_raw_os_error(-result));
        }
        Ok(result as usize)
    }
}

impl super::StorageEngine for IoUringStorage {
    const KIND: StorageKind = StorageKind::IoUring;

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        crate::storage::availability::StorageCapabilities::cached()
            .io_uring
            .clone()
    }
}

impl super::ReadableStorage for IoUringStorage {
    fn read_file(&self, path: &Path) -> IoResult<OwnedBytes> {
        Self::ensure_available()?;
        let file = OpenOptions::new().read(true).open(path)?;
        let len =
            usize::try_from(file.metadata()?.len()).map_err(|_| Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut buf = vec![0u8; len];
        Self::read_exact_at(&file, 0, &mut buf)?;
        Ok(OwnedBytes::Vec(buf))
    }

    fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes> {
        if range.is_empty() {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        Self::ensure_available()?;
        let file = OpenOptions::new().read(true).open(path)?;
        let mut buf = vec![0u8; range.len_usize()?];
        Self::read_exact_at(&file, range.start(), &mut buf)?;
        Ok(OwnedBytes::Vec(buf))
    }

    fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>> {
        use rayon::prelude::*;
        ranges
            .par_iter()
            .enumerate()
            .map(|(request_index, entry)| {
                let bytes = self.read_range(entry.path, entry.range)?.into_shared();
                Ok(RangeRead {
                    request_index,
                    range: entry.range,
                    bytes,
                })
            })
            .collect()
    }
}

impl super::WritableStorage for IoUringStorage {
    fn write_all_at(&self, file: &File, offset: u64, mut data: &[u8]) -> IoResult<()> {
        Self::ensure_available()?;
        let mut ring = IoUring::new(RING_DEPTH)?;
        let mut absolute_offset = offset;

        while !data.is_empty() {
            let chunk_len = data.len().min(MAX_IO_LEN);
            let (chunk, rest) = data.split_at(chunk_len);
            let mut written = 0usize;

            while written < chunk.len() {
                let len = (chunk.len() - written).min(MAX_IO_LEN);
                // SAFETY: `written < chunk.len()` and `len` is bounded by the
                // remaining bytes, so the resulting pointer stays within
                // `chunk` for the submitted write.
                let ptr = unsafe { chunk.as_ptr().add(written) };
                let entry = opcode::Write::new(types::Fd(file.as_raw_fd()), ptr, len as u32)
                    .offset(absolute_offset + written as u64)
                    .build()
                    .user_data(0);

                Self::submit_one(&mut ring, &entry)?;
                let result = Self::wait_one(&mut ring)?;
                if result == 0 {
                    return Err(Error::new(ErrorKind::WriteZero, "short io_uring write"));
                }
                written += result;
            }

            absolute_offset = absolute_offset
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "write offset overflow"))?;
            data = rest;
        }

        Ok(())
    }

    fn sync_data(&self, file: &File) -> IoResult<()> {
        Self::ensure_available()?;
        file.sync_data()
    }

    fn sync_all(&self, file: &File) -> IoResult<()> {
        Self::ensure_available()?;
        file.sync_all()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        ByteRange, FileRange, MappableStorage, ReadableStorage, StorageEngine, WritableStorage,
    };
    use tempfile::TempDir;

    fn write_tmp(dir: &TempDir, name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, data).unwrap();
        path
    }

    fn skip_if_unavailable() -> bool {
        !IoUringStorage::availability().is_available()
    }

    #[test]
    fn kind_is_io_uring() {
        assert_eq!(IoUringStorage::new().kind(), StorageKind::IoUring);
    }

    #[test]
    fn availability_reflects_kernel() {
        // Just assert it returns without panic — value depends on the host.
        let _ = IoUringStorage::availability();
    }

    #[test]
    fn read_file_roundtrip() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let path = write_tmp(&dir, "file.bin", &data);

        let result = IoUringStorage::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[test]
    fn read_file_empty() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "empty.bin", b"");

        let result = IoUringStorage::new().read_file(&path).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_range_returns_correct_slice() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..100).collect();
        let path = write_tmp(&dir, "range.bin", &data);

        let result = IoUringStorage::new()
            .read_range(&path, ByteRange::from_offset_len(10, 20).unwrap())
            .unwrap();
        assert_eq!(result.as_ref(), &data[10..30]);
    }

    #[test]
    fn read_range_zero_len() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "z.bin", b"hello");

        let result = IoUringStorage::new()
            .read_range(&path, ByteRange::from_offset_len(0, 0).unwrap())
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_ranges_empty() {
        if skip_if_unavailable() {
            return;
        }
        let results = IoUringStorage::new().read_ranges(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn read_ranges_single() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..200).collect();
        let path = write_tmp(&dir, "batch.bin", &data);

        let entries = [FileRange::new(
            &path,
            ByteRange::from_offset_len(50, 30).unwrap(),
        )];
        let results = IoUringStorage::new().read_ranges(&entries).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data(), &data[50..80]);
    }

    #[test]
    fn read_ranges_multiple_preserves_order() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        let path = write_tmp(&dir, "multi.bin", &data);

        let entries = [
            FileRange::new(&path, ByteRange::from_offset_len(0, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(20, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(100, 5).unwrap()),
        ];
        let results = IoUringStorage::new().read_ranges(&entries).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data(), &data[0..10]);
        assert_eq!(results[1].data(), &data[20..30]);
        assert_eq!(results[2].data(), &data[100..105]);
    }

    #[test]
    fn write_and_read_roundtrip() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let data = b"hello io_uring write";
        {
            let file = std::fs::File::create(&path).unwrap();
            IoUringStorage::new().write_all_at(&file, 0, data).unwrap();
            IoUringStorage::new().sync_all(&file).unwrap();
        }
        let result = IoUringStorage::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), data);
    }
}
