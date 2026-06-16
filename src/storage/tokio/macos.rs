//! macOS Tokio async storage implementation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::storage::{
    AsyncReadableStorage, AsyncWritableStorage, ByteRange, FileRange, IoResult, RangeRead,
    ReadableStorage, WritableStorage, WriteSlice,
    availability::{StorageAvailability, StorageKind},
    buffer::OwnedBytes,
    sync::SyncStorage,
};

/// Tokio async storage engine (macOS implementation).
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioStorage;

impl TokioStorage {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
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
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || SyncStorage::new().read_file(&path))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    async fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || SyncStorage::new().read_range(&path, range))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    async fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        let owned: Vec<(PathBuf, ByteRange)> = ranges
            .iter()
            .map(|e| (e.path.to_path_buf(), e.range))
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
            results.push(RangeRead { request_index, range, bytes });
        }
        Ok(results)
    }
}

impl AsyncWritableStorage for TokioStorage {
    async fn write_file(&self, path: &Path, data: &[u8]) -> IoResult<()> {
        let path = path.to_path_buf();
        let data = data.to_vec();
        tokio::task::spawn_blocking(move || SyncStorage::new().write_file(&path, &data))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    async fn write_positioned_file(
        &self,
        path: &Path,
        len: u64,
        writes: &[WriteSlice<'_>],
    ) -> IoResult<()> {
        let path = path.to_path_buf();
        let owned: Vec<(u64, Vec<u8>)> =
            writes.iter().map(|w| (w.offset, w.data.to_vec())).collect();
        tokio::task::spawn_blocking(move || {
            let slices: Vec<WriteSlice<'_>> =
                owned.iter().map(|(o, d)| WriteSlice::new(*o, d)).collect();
            SyncStorage::new().write_positioned_file(&path, len, &slices)
        })
        .await
        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    async fn write_at(&self, path: &Path, offset: u64, data: &[u8]) -> IoResult<()> {
        let path = path.to_path_buf();
        let data = data.to_vec();
        tokio::task::spawn_blocking(move || SyncStorage::new().write_at(&path, offset, &data))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    async fn write_slices(&self, path: &Path, writes: &[WriteSlice<'_>]) -> IoResult<()> {
        let path = path.to_path_buf();
        let owned: Vec<(u64, Vec<u8>)> =
            writes.iter().map(|w| (w.offset, w.data.to_vec())).collect();
        tokio::task::spawn_blocking(move || {
            let slices: Vec<WriteSlice<'_>> =
                owned.iter().map(|(o, d)| WriteSlice::new(*o, d)).collect();
            SyncStorage::new().write_slices(&path, &slices)
        })
        .await
        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    async fn sync_data(&self, path: &Path) -> IoResult<()> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || SyncStorage::new().sync_data(&path))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    async fn sync_all(&self, path: &Path) -> IoResult<()> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || SyncStorage::new().sync_all(&path))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }
}
