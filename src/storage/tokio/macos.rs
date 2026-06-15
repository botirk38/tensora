//! macOS Tokio async storage implementation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::storage::{
    AsyncReadableStorage, AsyncWritableStorage, ByteRange, FileRange, IoResult, RangeRead,
    ReadableStorage, WriteMode, WriteOptions,
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

/// A file handle opened for async writes.
pub struct TokioWriter {
    file: tokio::fs::File,
}

impl std::fmt::Debug for TokioWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokioWriter").finish_non_exhaustive()
    }
}

impl TokioWriter {
    pub async fn create(path: &Path, options: WriteOptions) -> IoResult<Self> {
        if options.create_parent_dirs
            && let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut open = tokio::fs::OpenOptions::new();
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
        let file = open.open(path).await?;
        if let Some(len) = options.preallocate {
            file.set_len(len).await?;
        }
        Ok(Self { file })
    }

    pub async fn write_all_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()> {
        use tokio::io::{AsyncSeekExt, AsyncWriteExt};
        self.file.seek(std::io::SeekFrom::Start(offset)).await?;
        self.file.write_all(data).await
    }

    pub async fn sync_data(&mut self) -> IoResult<()> {
        self.file.sync_data().await
    }

    pub async fn sync_all(&mut self) -> IoResult<()> {
        self.file.sync_all().await
    }
}

impl super::super::StorageEngine for TokioWriter {
    const KIND: StorageKind = StorageKind::Tokio;

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        StorageAvailability::Available
    }
}

impl AsyncWritableStorage for TokioWriter {
    async fn write_all_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()> {
        TokioWriter::write_all_at(self, offset, data).await
    }

    async fn set_len(&mut self, len: u64) -> IoResult<()> {
        self.file.set_len(len).await
    }

    async fn sync_data(&mut self) -> IoResult<()> {
        TokioWriter::sync_data(self).await
    }

    async fn sync_all(&mut self) -> IoResult<()> {
        TokioWriter::sync_all(self).await
    }
}
