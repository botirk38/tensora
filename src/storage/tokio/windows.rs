//! Windows Tokio async storage implementation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::storage::{
    AsyncReadableStorage, AsyncWritableStorage, ByteRange, FileRange, IoResult, RangeRead,
    availability::{StorageAvailability, StorageKind},
    buffer::OwnedBytes,
    sync::SyncStorage,
};

/// Tokio async storage engine (Windows implementation).
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioStorage;

impl TokioStorage {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub async fn read_file(&self, path: &Path) -> IoResult<OwnedBytes> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || SyncStorage::new().read_file(&path))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    pub async fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || SyncStorage::new().read_range(&path, range))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    pub async fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }

        let owned: Vec<(PathBuf, ByteRange)> = ranges
            .iter()
            .map(|entry| (entry.path.to_path_buf(), entry.range))
            .collect();
        let handles: Vec<_> = owned
            .into_iter()
            .map(|(path, range)| {
                tokio::task::spawn_blocking(move || {
                    let bytes = SyncStorage::new().read_range(&path, range)?.into_shared();
                    Ok::<_, std::io::Error>((range, bytes))
                })
            })
            .collect();

        let mut results = Vec::with_capacity(handles.len());
        for (request_index, handle) in handles.into_iter().enumerate() {
            let (range, bytes): (ByteRange, Arc<[u8]>) = handle
                .await
                .map_err(|_| std::io::Error::other("spawn_blocking panicked"))??;
            results.push(RangeRead {
                request_index,
                range,
                bytes,
            });
        }
        Ok(results)
    }
}

impl super::super::StorageEngine for TokioStorage {
    const KIND: StorageKind = StorageKind::Tokio;

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        StorageAvailability::Available
    }
}

impl AsyncReadableStorage for TokioStorage {
    async fn read_file(&self, path: &Path) -> IoResult<OwnedBytes> {
        TokioStorage::read_file(self, path).await
    }

    async fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes> {
        TokioStorage::read_range(self, path, range).await
    }

    async fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>> {
        TokioStorage::read_ranges(self, ranges).await
    }
}

impl AsyncWritableStorage for TokioStorage {
    async fn write_all_at(&self, file: &std::fs::File, offset: u64, data: &[u8]) -> IoResult<()> {
        let file = file.try_clone()?;
        let data = data.to_vec();
        tokio::task::spawn_blocking(move || {
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
        })
        .await
        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    async fn sync_data(&self, file: &std::fs::File) -> IoResult<()> {
        let file = file.try_clone()?;
        tokio::task::spawn_blocking(move || file.sync_data())
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    async fn sync_all(&self, file: &std::fs::File) -> IoResult<()> {
        let file = file.try_clone()?;
        tokio::task::spawn_blocking(move || file.sync_all())
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{AsyncReadableStorage, AsyncWritableStorage, ByteRange, FileRange};
    use crate::test_utils::run_async;
    use tempfile::TempDir;

    fn write_tmp(dir: &TempDir, name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn read_file() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let path = write_tmp(&dir, "file.bin", &data);

        let result = run_async(TokioStorage::new().read_file(&path)).unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[test]
    fn read_range() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..100).collect();
        let path = write_tmp(&dir, "range.bin", &data);

        let result = run_async(
            TokioStorage::new().read_range(&path, ByteRange::from_offset_len(10, 20).unwrap()),
        )
        .unwrap();
        assert_eq!(result.as_ref(), &data[10..30]);
    }

    #[test]
    fn read_ranges() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        let path = write_tmp(&dir, "multi.bin", &data);

        let entries = [
            FileRange::new(&path, ByteRange::from_offset_len(0, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(20, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(100, 5).unwrap()),
        ];
        let results = run_async(TokioStorage::new().read_ranges(&entries)).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data(), &data[0..10]);
        assert_eq!(results[1].data(), &data[20..30]);
        assert_eq!(results[2].data(), &data[100..105]);
    }

    #[test]
    fn write_and_read_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let data = b"hello windows tokio";

        let file = std::fs::File::create(&path).unwrap();
        run_async(TokioStorage::new().write_all_at(&file, 0, data)).unwrap();
        run_async(TokioStorage::new().sync_all(&file)).unwrap();
        drop(file);

        let result = run_async(TokioStorage::new().read_file(&path)).unwrap();
        assert_eq!(result.as_ref(), data);
    }
}
