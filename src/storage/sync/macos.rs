//! macOS std::fs synchronous storage implementation.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use crate::storage::{
    ByteRange, FileRange, IoResult, RangeRead, WriteMode, WriteOptions,
    availability::{StorageAvailability, StorageKind},
    buffer::{OwnedBytes, get_buffer_pool},
};

// ============================================================================
// SyncStorage
// ============================================================================

/// Synchronous blocking storage engine (macOS std::fs implementation).
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

    fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes> {
        if range.is_empty() {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut file = std::fs::File::open(path)?;
        file.seek(SeekFrom::Start(range.start()))?;
        let mut buf = get_buffer_pool().get(range.len_usize()?);
        file.read_exact(&mut buf[..])?;
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

// ============================================================================
// SyncWriter
// ============================================================================

/// A file handle opened for synchronous writes.
pub struct SyncWriter {
    file: std::fs::File,
}

impl SyncWriter {
    pub fn create(path: &Path, options: WriteOptions) -> IoResult<Self> {
        if options.create_parent_dirs
            && let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut open = std::fs::OpenOptions::new();
        open.write(true);
        match options.mode {
            WriteMode::CreateNew => {
                open.create_new(true);
            }
            WriteMode::CreateOrTruncate => {
                open.create(true).truncate(true);
            }
            WriteMode::OpenExisting => {}
        }
        let file = open.open(path)?;
        if let Some(len) = options.preallocate {
            file.set_len(len)?;
        }
        Ok(Self { file })
    }
}

impl std::fmt::Debug for SyncWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncWriter").finish_non_exhaustive()
    }
}

impl super::super::StorageEngine for SyncWriter {
    const KIND: StorageKind = StorageKind::Sync;

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        StorageAvailability::Available
    }
}

impl super::super::WritableStorage for SyncWriter {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(data)
    }

    fn set_len(&mut self, len: u64) -> IoResult<()> {
        self.file.set_len(len)
    }

    fn sync_data(&mut self) -> IoResult<()> {
        self.file.sync_data()
    }

    fn sync_all(&mut self) -> IoResult<()> {
        self.file.sync_all()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        ByteRange, FileRange, ReadableStorage, StorageEngine, WritableStorage, WriteOptions,
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
    fn write_at_and_flush_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let mut writer = SyncWriter::create(&path, WriteOptions::create_or_truncate()).unwrap();
        writer.write_all_at(0, b"hello world").unwrap();
        writer.sync_all().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello world");
    }
}
