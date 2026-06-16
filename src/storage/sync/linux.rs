//! Linux O_DIRECT-aware synchronous storage implementation.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use crate::storage::{
    ByteRange, FileRange, IoResult, RangeRead,
    availability::{StorageAvailability, StorageKind},
    buffer::{AlignedBuffer, OwnedBytes, get_buffer_pool},
};

const BLOCK_SIZE: usize = 4096;
const BLOCK_SIZE_U64: u64 = 4096;
const MAX_SINGLE_READ: usize = 512 * 1024 * 1024;

/// Wrapper around a raw mutable pointer to a disjoint buffer slice so we can
/// move it into `std::thread::spawn`. The caller must guarantee that no two
/// `SendSlice` values alias the same memory and that all threads are joined
/// before the backing buffer is accessed or dropped.
struct SendSlice(*mut u8, usize);
// SAFETY: see per-callsite comments in `load_chunked`.
unsafe impl Send for SendSlice {}

// ============================================================================
// SyncStorage
// ============================================================================

/// Synchronous blocking storage engine (Linux O_DIRECT implementation).
#[derive(Debug, Clone, Copy, Default)]
pub struct SyncStorage;

impl SyncStorage {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn round_up_to_block(n: usize) -> usize {
        (n + BLOCK_SIZE - 1) & !(BLOCK_SIZE - 1)
    }

    fn open_prefer_direct(path: &Path) -> IoResult<(std::fs::File, bool)> {
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

    fn read_direct(file: &mut std::fs::File, buf: &mut [u8], actual_len: usize) -> IoResult<()> {
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

    fn load_chunked(path: &Path, chunks: usize) -> IoResult<OwnedBytes> {
        let (file, direct) = Self::open_prefer_direct(path)?;
        let file_size = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        let raw_chunk = file_size.div_ceil(chunks).max(1);
        let chunk_size = if direct {
            Self::round_up_to_block(raw_chunk)
        } else {
            raw_chunk
        };
        drop(file);

        if direct {
            let aligned_total = Self::round_up_to_block(file_size);
            let mut final_buf = AlignedBuffer::new(aligned_total)?;
            final_buf.set_len(aligned_total);

            // Collect (ptr, actual_len, start, path) tuples while we hold &mut
            // final_buf, then drop the borrow before spawning threads.
            // SAFETY: each SendSlice points to a distinct, non-overlapping
            // region of final_buf; all threads are joined before final_buf is
            // accessed or dropped.
            let tasks: Vec<(SendSlice, usize, u64, std::path::PathBuf)> = (0..chunks)
                .filter_map(|i| {
                    let start = i * chunk_size;
                    let end = std::cmp::min(start + chunk_size, file_size);
                    if start >= end {
                        return None;
                    }
                    let read_len = Self::round_up_to_block(end - start);
                    let buf_end = std::cmp::min(start + read_len, aligned_total);
                    let slice = final_buf.as_mut_slice().get_mut(start..buf_end)?;
                    Some((
                        SendSlice(slice.as_mut_ptr(), slice.len()),
                        end - start,
                        start as u64,
                        path.to_path_buf(),
                    ))
                })
                .collect();

            let handles: Vec<_> = tasks
                .into_iter()
                .map(|(ptr, actual_len, start_off, path_clone)| {
                    std::thread::spawn(move || {
                        let buf = unsafe { std::slice::from_raw_parts_mut(ptr.0, ptr.1) };
                        let (mut f, _) = Self::open_prefer_direct(&path_clone)?;
                        f.seek(SeekFrom::Start(start_off))?;
                        Self::read_direct(&mut f, buf, actual_len)?;
                        std::io::Result::Ok(())
                    })
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

            // Same two-phase approach: collect SendSlice handles first, then spawn.
            // SAFETY: each SendSlice points to a distinct, non-overlapping
            // region of final_buf; all threads are joined before final_buf is
            // accessed or dropped.
            let tasks: Vec<(SendSlice, u64, std::path::PathBuf)> = (0..chunks)
                .filter_map(|i| {
                    let start = i * chunk_size;
                    let end = std::cmp::min(start + chunk_size, file_size);
                    if start >= end {
                        return None;
                    }
                    let slice = final_buf.as_mut_slice().get_mut(start..end)?;
                    Some((
                        SendSlice(slice.as_mut_ptr(), slice.len()),
                        start as u64,
                        path.to_path_buf(),
                    ))
                })
                .collect();

            let handles: Vec<_> = tasks
                .into_iter()
                .map(|(ptr, start_off, path_clone)| {
                    std::thread::spawn(move || {
                        let buf = unsafe { std::slice::from_raw_parts_mut(ptr.0, ptr.1) };
                        let mut f = std::fs::File::open(&path_clone)?;
                        f.seek(SeekFrom::Start(start_off))?;
                        f.read_exact(buf)?;
                        std::io::Result::Ok(())
                    })
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
}

impl super::super::StorageEngine for SyncStorage {
    const KIND: StorageKind = StorageKind::Sync;

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        StorageAvailability::Available
    }
}

impl super::super::ReadableStorage for SyncStorage {
    fn read_file(&self, path: &Path) -> IoResult<OwnedBytes> {
        let (mut file, direct) = Self::open_prefer_direct(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        if len > MAX_SINGLE_READ {
            let chunks = (len / (128 * 1024 * 1024)).clamp(1, 8);
            return Self::load_chunked(path, chunks);
        }
        if direct {
            let aligned_len = Self::round_up_to_block(len);
            let mut buf = AlignedBuffer::new(aligned_len)?;
            buf.set_len(aligned_len);
            Self::read_direct(&mut file, buf.as_mut_slice(), len)?;
            buf.set_len(len);
            Ok(OwnedBytes::Aligned(buf))
        } else {
            let mut buf = get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(OwnedBytes::Pooled(buf))
        }
    }

    fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes> {
        if range.is_empty() {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let len = range.len_usize()?;
        let offset = range.start();
        let (mut file, direct) = Self::open_prefer_direct(path)?;
        if direct {
            let aligned_offset = offset & !(BLOCK_SIZE_U64 - 1);
            let head_skip = (offset - aligned_offset) as usize;
            let aligned_len = Self::round_up_to_block(head_skip + len);
            file.seek(SeekFrom::Start(aligned_offset))?;
            let mut buf = AlignedBuffer::new(aligned_len)?;
            buf.set_len(aligned_len);
            Self::read_direct(&mut file, buf.as_mut_slice(), head_skip + len)?;
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

impl super::super::WritableStorage for SyncStorage {
    fn write_all_at(&self, file: &std::fs::File, offset: u64, data: &[u8]) -> IoResult<()> {
        use std::os::unix::fs::FileExt;
        let mut written = 0usize;
        while written < data.len() {
            let n = file.write_at(&data[written..], offset + written as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "write_at returned zero bytes",
                ));
            }
            written += n;
        }
        Ok(())
    }

    fn sync_data(&self, file: &std::fs::File) -> IoResult<()> {
        file.sync_data()
    }

    fn sync_all(&self, file: &std::fs::File) -> IoResult<()> {
        file.sync_all()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{ByteRange, FileRange, ReadableStorage, StorageEngine, WritableStorage};
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

        let result = SyncStorage::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[test]
    fn read_file_empty() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "empty.bin", b"");

        let result = SyncStorage::new().read_file(&path).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_range_returns_correct_slice() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..100).collect();
        let path = write_tmp(&dir, "range.bin", &data);

        let result = SyncStorage::new()
            .read_range(&path, ByteRange::from_offset_len(10, 20).unwrap())
            .unwrap();
        assert_eq!(result.as_ref(), &data[10..30]);
    }

    #[test]
    fn read_range_zero_len() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "z.bin", b"hello");

        let result = SyncStorage::new()
            .read_range(&path, ByteRange::from_offset_len(0, 0).unwrap())
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_ranges_empty() {
        let results = SyncStorage::new().read_ranges(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn read_ranges_single() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..200).collect();
        let path = write_tmp(&dir, "batch.bin", &data);

        let entries = [FileRange::new(
            &path,
            ByteRange::from_offset_len(50, 30).unwrap(),
        )];
        let results = SyncStorage::new().read_ranges(&entries).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data(), &data[50..80]);
    }

    #[test]
    fn read_ranges_multiple_preserves_order() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        let path = write_tmp(&dir, "multi.bin", &data);

        let entries = [
            FileRange::new(&path, ByteRange::from_offset_len(0, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(20, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(100, 5).unwrap()),
        ];
        let results = SyncStorage::new().read_ranges(&entries).unwrap();

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
    fn write_and_read_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let data = b"hello linux sync";
        let file = std::fs::File::create(&path).unwrap();
        SyncStorage::new().write_all_at(&file, 0, data).unwrap();
        SyncStorage::new().sync_all(&file).unwrap();
        drop(file);
        let result = SyncStorage::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), data);
    }
}
