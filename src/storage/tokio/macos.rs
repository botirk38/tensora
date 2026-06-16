//! macOS Tokio async storage implementation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::storage::{
    AsyncReadableStorage, AsyncWritableStorage, ByteRange, FileRange, IoResult, RangeRead,
    ReadableStorage,
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
