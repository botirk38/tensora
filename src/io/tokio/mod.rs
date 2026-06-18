//! Tokio async I/O backend.
//!
//! [`Tokio`] implements [`AsyncIo`] using direct `tokio::fs` I/O.
//! Batch methods use bounded concurrency configured by [`TokioOptions`].
//!
//! [`AsyncIo`]: crate::io::AsyncIo

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ::tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use futures::StreamExt;

use crate::io::{
    AsyncIo, ByteRange, FileRange, IoResult, RangeRead, RequestIndex, WriteSlices,
    availability::{IoAvailability, IoKind},
    buffer::OwnedBytes,
};

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
compile_error!("tensora io::tokio supports Linux, macOS, and Windows only");

const DEFAULT_BATCH_CONCURRENCY: usize = 64;

/// Options for the Tokio async I/O backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokioOptions {
    /// Maximum number of concurrent tasks for batch reads/writes.
    pub batch_concurrency: usize,
}

impl Default for TokioOptions {
    fn default() -> Self {
        Self {
            batch_concurrency: DEFAULT_BATCH_CONCURRENCY,
        }
    }
}

/// Tokio async I/O backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tokio {
    options: TokioOptions,
}

impl Tokio {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            options: TokioOptions {
                batch_concurrency: DEFAULT_BATCH_CONCURRENCY,
            },
        }
    }

    pub fn with_options(options: TokioOptions) -> IoResult<Self> {
        if options.batch_concurrency == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "batch_concurrency must be greater than zero",
            ));
        }
        Ok(Self { options })
    }

    #[inline]
    #[must_use]
    pub const fn options(&self) -> &TokioOptions {
        &self.options
    }
}

impl super::Io for Tokio {
    const KIND: IoKind = IoKind::Tokio;

    fn availability() -> IoAvailability
    where
        Self: Sized,
    {
        IoAvailability::Available
    }
}

impl AsyncIo for Tokio {
    async fn read_file(&self, path: &Path) -> IoResult<OwnedBytes> {
        let bytes = ::tokio::fs::read(path).await?;
        Ok(OwnedBytes::Vec(bytes))
    }

    async fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes> {
        if range.is_empty() {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut file = ::tokio::fs::File::open(path).await?;
        file.seek(std::io::SeekFrom::Start(range.start())).await?;
        let mut buf = vec![0u8; range.len_usize()?];
        file.read_exact(&mut buf).await?;
        Ok(OwnedBytes::Vec(buf))
    }

    async fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        let tasks: Vec<(RequestIndex, PathBuf, ByteRange)> = ranges
            .iter()
            .enumerate()
            .map(|(i, e)| (RequestIndex::new(i), e.path.to_path_buf(), e.range))
            .collect();

        let stream = futures::stream::iter(tasks).map(|(request_index, path, range)| async move {
            let mut file = ::tokio::fs::File::open(&path).await?;
            file.seek(std::io::SeekFrom::Start(range.start())).await?;
            let mut buf = vec![0u8; range.len_usize()?];
            file.read_exact(&mut buf).await?;
            let bytes: Arc<[u8]> = Arc::from(buf);
            Ok::<RangeRead, std::io::Error>(RangeRead {
                request_index,
                range,
                bytes,
            })
        });

        let mut results: Vec<RangeRead> = stream
            .buffer_unordered(self.options.batch_concurrency)
            .collect::<Vec<IoResult<RangeRead>>>()
            .await
            .into_iter()
            .collect::<IoResult<Vec<RangeRead>>>()?;
        results.sort_unstable_by_key(|r| r.request_index);
        Ok(results)
    }

    async fn write_file(&self, path: &Path, data: &[u8]) -> IoResult<()> {
        ::tokio::fs::write(path, data).await
    }

    async fn write_positioned_file(
        &self,
        path: &Path,
        len: u64,
        writes: WriteSlices<'_>,
    ) -> IoResult<()> {
        {
            let file = ::tokio::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
                .await?;
            file.set_len(len).await?;
        }
        if writes.is_empty() {
            return Ok(());
        }
        let tasks: Vec<(PathBuf, u64, Vec<u8>)> = writes
            .as_slice()
            .iter()
            .map(|w| (path.to_path_buf(), w.offset, w.data.to_vec()))
            .collect();

        futures::stream::iter(tasks)
            .map(|(path, offset, data)| async move {
                let mut file = ::tokio::fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .await?;
                file.seek(std::io::SeekFrom::Start(offset)).await?;
                file.write_all(&data).await
            })
            .buffer_unordered(self.options.batch_concurrency)
            .collect::<Vec<IoResult<()>>>()
            .await
            .into_iter()
            .collect()
    }

    async fn write_at(&self, path: &Path, offset: u64, data: &[u8]) -> IoResult<()> {
        let mut file = ::tokio::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .await?;
        file.seek(std::io::SeekFrom::Start(offset)).await?;
        file.write_all(data).await
    }

    async fn write_slices(&self, path: &Path, writes: WriteSlices<'_>) -> IoResult<()> {
        if writes.is_empty() {
            return Ok(());
        }
        let tasks: Vec<(PathBuf, u64, Vec<u8>)> = writes
            .as_slice()
            .iter()
            .map(|w| (path.to_path_buf(), w.offset, w.data.to_vec()))
            .collect();

        futures::stream::iter(tasks)
            .map(|(path, offset, data)| async move {
                let mut file = ::tokio::fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .await?;
                file.seek(std::io::SeekFrom::Start(offset)).await?;
                file.write_all(&data).await
            })
            .buffer_unordered(self.options.batch_concurrency)
            .collect::<Vec<IoResult<()>>>()
            .await
            .into_iter()
            .collect()
    }

    async fn sync_data(&self, path: &Path) -> IoResult<()> {
        ::tokio::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .await?
            .sync_data()
            .await
    }

    async fn sync_all(&self, path: &Path) -> IoResult<()> {
        ::tokio::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .await?
            .sync_all()
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{AsyncIo, ByteRange, FileRange, WriteSlice, WriteSlices};
    use crate::test_utils::run_async;
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
        let result = run_async(Tokio::new().read_file(&path)).unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[test]
    fn read_file_empty() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "empty.bin", b"");
        let result = run_async(Tokio::new().read_file(&path)).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_range_returns_correct_slice() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..100).collect();
        let path = write_tmp(&dir, "range.bin", &data);
        let result =
            run_async(Tokio::new().read_range(&path, ByteRange::from_offset_len(10, 20).unwrap()))
                .unwrap();
        assert_eq!(result.as_ref(), &data[10..30]);
    }

    #[test]
    fn read_range_zero_len() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "z.bin", b"hello");
        let result =
            run_async(Tokio::new().read_range(&path, ByteRange::from_offset_len(0, 0).unwrap()))
                .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_ranges_empty() {
        let results = run_async(Tokio::new().read_ranges(&[])).unwrap();
        assert!(results.is_empty());
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
        let results = run_async(Tokio::new().read_ranges(&entries)).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data(), &data[0..10]);
        assert_eq!(results[1].data(), &data[20..30]);
        assert_eq!(results[2].data(), &data[100..105]);
    }

    #[test]
    fn write_file_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let data = b"hello tokio";
        run_async(Tokio::new().write_file(&path, data)).unwrap();
        run_async(Tokio::new().sync_all(&path)).unwrap();
        let result = run_async(Tokio::new().read_file(&path)).unwrap();
        assert_eq!(result.as_ref(), data);
    }

    #[test]
    fn write_file_truncates_existing() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "trunc.bin", b"old content here");
        run_async(Tokio::new().write_file(&path, b"new")).unwrap();
        let result = run_async(Tokio::new().read_file(&path)).unwrap();
        assert_eq!(result.as_ref(), b"new");
    }

    #[test]
    fn write_at_preserves_surrounding_bytes() {
        let dir = TempDir::new().unwrap();
        let mut data = b"AAABBBCCC".to_vec();
        let path = write_tmp(&dir, "patch.bin", &data);
        run_async(Tokio::new().write_at(&path, 3, b"XXX")).unwrap();
        data[3..6].copy_from_slice(b"XXX");
        let result = run_async(Tokio::new().read_file(&path)).unwrap();
        assert_eq!(result.as_ref(), &data);
    }

    #[test]
    fn write_positioned_file_creates_exact_length() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pos.bin");
        let writes = [WriteSlice::new(0, b"HELLO"), WriteSlice::new(10, b"WORLD")];
        run_async(Tokio::new().write_positioned_file(
            &path,
            15,
            WriteSlices::new(&writes).unwrap(),
        ))
        .unwrap();
        let result = run_async(Tokio::new().read_file(&path)).unwrap();
        assert_eq!(result.len(), 15);
        assert_eq!(&result.as_ref()[0..5], b"HELLO");
        assert_eq!(&result.as_ref()[10..15], b"WORLD");
    }

    #[test]
    fn write_positioned_file_empty_batch_creates_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty_pos.bin");
        run_async(Tokio::new().write_positioned_file(&path, 16, WriteSlices::new(&[]).unwrap()))
            .unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 16);
    }

    #[test]
    fn write_slices_batches_into_existing_file() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "batch_write.bin", &[0u8; 20]);
        let writes = [WriteSlice::new(0, b"HELLO"), WriteSlice::new(15, b"WORLD")];
        run_async(Tokio::new().write_slices(&path, WriteSlices::new(&writes).unwrap())).unwrap();
        let result = run_async(Tokio::new().read_file(&path)).unwrap();
        assert_eq!(&result.as_ref()[0..5], b"HELLO");
        assert_eq!(&result.as_ref()[15..20], b"WORLD");
    }

    #[test]
    fn write_slices_empty_batch_is_noop() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "noop.bin", b"unchanged");
        run_async(Tokio::new().write_slices(&path, WriteSlices::new(&[]).unwrap())).unwrap();
        let result = run_async(Tokio::new().read_file(&path)).unwrap();
        assert_eq!(result.as_ref(), b"unchanged");
    }
}
