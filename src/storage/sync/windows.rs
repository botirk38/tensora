//! Windows std::fs synchronous storage implementation.

use std::io::Write;
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

    fn open_create_truncate(path: &Path) -> IoResult<std::fs::File> {
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
    }

    fn open_write_existing(path: &Path) -> IoResult<std::fs::File> {
        std::fs::OpenOptions::new().write(true).open(path)
    }

    fn write_all_at_file(file: &std::fs::File, offset: u64, data: &[u8]) -> IoResult<()> {
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
    fn write_file(&self, path: &Path, data: &[u8]) -> IoResult<()> {
        let mut file = Self::open_create_truncate(path)?;
        file.write_all(data)
    }

    fn write_positioned_file(
        &self,
        path: &Path,
        len: u64,
        writes: &[WriteSlice<'_>],
    ) -> IoResult<()> {
        let file = Self::open_create_truncate(path)?;
        file.set_len(len)?;
        for w in writes {
            Self::write_all_at_file(&file, w.offset, w.data)?;
        }
        Ok(())
    }

    fn write_at(&self, path: &Path, offset: u64, data: &[u8]) -> IoResult<()> {
        let file = Self::open_write_existing(path)?;
        Self::write_all_at_file(&file, offset, data)
    }

    fn write_slices(&self, path: &Path, writes: &[WriteSlice<'_>]) -> IoResult<()> {
        let file = Self::open_write_existing(path)?;
        for w in writes {
            Self::write_all_at_file(&file, w.offset, w.data)?;
        }
        Ok(())
    }

    fn sync_data(&self, path: &Path) -> IoResult<()> {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)?
            .sync_data()
    }

    fn sync_all(&self, path: &Path) -> IoResult<()> {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)?
            .sync_all()
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
    fn write_file_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let data = b"hello windows sync";
        SyncStorage::new().write_file(&path, data).unwrap();
        SyncStorage::new().sync_all(&path).unwrap();
        let result = SyncStorage::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), data);
    }

    #[test]
    fn write_file_truncates_existing() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "trunc.bin", b"old content here");
        SyncStorage::new().write_file(&path, b"new").unwrap();
        let result = SyncStorage::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), b"new");
    }

    #[test]
    fn write_at_preserves_surrounding_bytes() {
        let dir = TempDir::new().unwrap();
        let mut data = b"AAABBBCCC".to_vec();
        let path = write_tmp(&dir, "patch.bin", &data);
        SyncStorage::new().write_at(&path, 3, b"XXX").unwrap();
        data[3..6].copy_from_slice(b"XXX");
        let result = SyncStorage::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), &data);
    }

    #[test]
    fn write_positioned_file_creates_exact_length() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pos.bin");
        let writes = [WriteSlice::new(0, b"HELLO"), WriteSlice::new(10, b"WORLD")];
        SyncStorage::new()
            .write_positioned_file(&path, 15, &writes)
            .unwrap();
        let result = SyncStorage::new().read_file(&path).unwrap();
        assert_eq!(result.len(), 15);
        assert_eq!(&result.as_ref()[0..5], b"HELLO");
        assert_eq!(&result.as_ref()[10..15], b"WORLD");
    }

    #[test]
    fn write_slices_batches_into_existing_file() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "batch_write.bin", &[0u8; 20]);
        let writes = [WriteSlice::new(0, b"HELLO"), WriteSlice::new(15, b"WORLD")];
        SyncStorage::new().write_slices(&path, &writes).unwrap();
        let result = SyncStorage::new().read_file(&path).unwrap();
        assert_eq!(&result.as_ref()[0..5], b"HELLO");
        assert_eq!(&result.as_ref()[15..20], b"WORLD");
    }
}
