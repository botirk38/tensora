//! Windows std::fs synchronous storage implementation.

use std::path::Path;
use std::sync::Arc;

use crate::storage::{
    ByteRange, FileRange, IoResult, RangeRead, WriteSlice,
    availability::{StorageAvailability, StorageKind},
    buffer::{OwnedBytes, get_buffer_pool},
};

// ============================================================================
// SyncStorage
// ============================================================================

/// Synchronous blocking storage engine (Windows std::fs implementation).
#[derive(Debug, Clone, Copy, Default)]
pub struct SyncStorage;

impl SyncStorage {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
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
        use std::os::windows::fs::FileExt;
        let file = std::fs::File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut buf = get_buffer_pool().get(len);
        let mut read = 0usize;
        while read < len {
            let n = file.seek_read(&mut buf[read..], read as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "seek_read returned zero bytes before end of file",
                ));
            }
            read += n;
        }
        Ok(OwnedBytes::Pooled(buf))
    }

    fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes> {
        use std::os::windows::fs::FileExt;
        if range.is_empty() {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let file = std::fs::File::open(path)?;
        let mut buf = get_buffer_pool().get(range.len_usize()?);
        let mut read = 0usize;
        let start_offset = range.start();
        let len = range.len_usize()?;
        while read < len {
            let n = file.seek_read(&mut buf[read..], start_offset + read as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "seek_read returned zero bytes before end of range",
                ));
            }
            read += n;
        }
        Ok(OwnedBytes::Pooled(buf))
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
        use std::os::windows::fs::FileExt;
        let mut written = 0usize;
        while written < data.len() {
            let n = file.seek_write(&data[written..], offset + written as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "seek_write returned zero bytes",
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
    fn write_and_read_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let data = b"hello windows sync";
        let file = std::fs::File::create(&path).unwrap();
        SyncStorage::new().write_all_at(&file, 0, data).unwrap();
        SyncStorage::new().sync_all(&file).unwrap();
        drop(file);
        let result = SyncStorage::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), data);
    }
}
