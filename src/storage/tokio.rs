//! Tokio async storage engine.
//!
//! [`TokioStorage`] provides inherent `async fn` methods for file and range
//! reads.  It does **not** implement the synchronous [`ReadableStorage`] trait;
//! callers use the inherent methods directly.
//!
//! Write access is obtained by calling [`TokioStorage::create_writer`], which
//! returns a [`TokioWriter`] bound to an open file.
//!
//! Internally, blocking I/O is offloaded to `tokio::task::spawn_blocking` so
//! that async tasks are never stalled on disk calls.  On Linux the blocking
//! reads prefer O_DIRECT to bypass the page cache.
//!
//! [`ReadableStorage`]: crate::storage::ReadableStorage

use std::path::Path;
use std::sync::Arc;

use crate::storage::{
    BatchReadRequest, FileReadRequest, IoResult, RangeReadRequest, RangeReadResult, WriteAtRequest,
    availability::{StorageAvailability, StorageCapabilities, StorageKind},
    buffer::{OwnedBytes, get_buffer_pool},
};

#[cfg(target_os = "linux")]
use crate::storage::buffer::AlignedBuffer;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

// ============================================================================
// O_DIRECT helpers (Linux only)
// ============================================================================

#[cfg(target_os = "linux")]
const BLOCK_SIZE: usize = 4096;
#[cfg(target_os = "linux")]
const BLOCK_SIZE_U64: u64 = 4096;

#[cfg(target_os = "linux")]
#[inline]
const fn round_up_to_block(n: usize) -> usize {
    (n + BLOCK_SIZE - 1) & !(BLOCK_SIZE - 1)
}

/// Try O_DIRECT first; fall back to regular open if the filesystem rejects it.
#[cfg(target_os = "linux")]
fn open_prefer_direct(path: &Path) -> IoResult<(std::fs::File, bool)> {
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    {
        Ok(f) => Ok((f, true)),
        Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
            let f = std::fs::File::open(path)?;
            Ok((f, false))
        }
        Err(e) => Err(e),
    }
}

/// Read exactly `actual_len` bytes into an aligned O_DIRECT buffer.
///
/// O_DIRECT reads may return short at EOF (the kernel reads aligned blocks but
/// the file may be shorter than `aligned_len`).  This loops until `actual_len`
/// bytes have been read.
#[cfg(target_os = "linux")]
fn read_direct(file: &mut std::fs::File, buf: &mut [u8], actual_len: usize) -> IoResult<()> {
    use std::io::Read;
    let mut pos = 0usize;
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

// ============================================================================
// Blocking I/O helpers
// ============================================================================

/// Read an entire file into an [`OwnedBytes`] buffer (blocking).
///
/// On Linux attempts O_DIRECT first and falls back to a regular buffered read
/// if the filesystem does not support it.  On other platforms always uses a
/// plain `std::fs::File` read into the global buffer pool.
fn load_blocking(path: &Path) -> IoResult<OwnedBytes> {
    #[cfg(target_os = "linux")]
    {
        use std::io::Read;

        let (mut file, direct) = open_prefer_direct(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        if direct {
            let aligned_len = round_up_to_block(len);
            let mut buf = AlignedBuffer::new(aligned_len)?;
            buf.set_len(aligned_len);
            read_direct(&mut file, buf.as_mut_slice(), len)?;
            buf.set_len(len);
            Ok(OwnedBytes::Aligned(buf))
        } else {
            let mut buf = get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(OwnedBytes::Pooled(buf))
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        use std::io::Read;

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
}

/// Read a byte range from a file into an [`OwnedBytes`] buffer (blocking).
///
/// On Linux attempts O_DIRECT first; when the offset is not block-aligned the
/// aligned super-range is read and the requested slice is extracted into a
/// shared `Arc<[u8]>` to avoid carrying the full aligned buffer.
fn load_range_blocking(path: &Path, offset: u64, len: usize) -> IoResult<OwnedBytes> {
    if len == 0 {
        return Ok(OwnedBytes::Shared(Arc::new([])));
    }

    #[cfg(target_os = "linux")]
    {
        use std::io::{Read, Seek, SeekFrom};

        let (mut file, direct) = open_prefer_direct(path)?;
        if direct {
            let aligned_offset = offset & !(BLOCK_SIZE_U64 - 1);
            let head_skip = (offset - aligned_offset) as usize;
            let aligned_len = round_up_to_block(head_skip + len);

            file.seek(SeekFrom::Start(aligned_offset))?;
            let mut buf = AlignedBuffer::new(aligned_len)?;
            buf.set_len(aligned_len);
            read_direct(&mut file, buf.as_mut_slice(), head_skip + len)?;

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

    #[cfg(not(target_os = "linux"))]
    {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = std::fs::File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(OwnedBytes::Pooled(buf))
    }
}

// ============================================================================
// TokioStorage
// ============================================================================

/// Tokio async storage engine.
///
/// ```rust,ignore
/// use tensora::storage::tokio::TokioStorage;
/// use tensora::storage::FileReadRequest;
/// use std::path::Path;
///
/// let engine = TokioStorage::default();
/// let bytes = engine.read_file(FileReadRequest::new(Path::new("model.safetensors"))).await?;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioStorage;

impl TokioStorage {
    /// Create a new `TokioStorage` engine.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Read an entire file into an [`OwnedBytes`] buffer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened or read.
    pub async fn read_file(&self, req: FileReadRequest<'_>) -> IoResult<OwnedBytes> {
        let path_buf = req.path.to_path_buf();
        tokio::task::spawn_blocking(move || load_blocking(&path_buf))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    /// Read a contiguous byte range from a file.
    ///
    /// Returns an empty buffer immediately when `req.len == 0`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened or the range cannot
    /// be read.
    pub async fn read_range(&self, req: RangeReadRequest<'_>) -> IoResult<OwnedBytes> {
        if req.len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let path_buf = req.path.to_path_buf();
        let offset = req.offset;
        let len = req.len;
        tokio::task::spawn_blocking(move || load_range_blocking(&path_buf, offset, len))
            .await
            .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    /// Read a batch of byte ranges, possibly from multiple files.
    ///
    /// Ranges are read concurrently via `spawn_blocking` tasks.  Results are
    /// returned in the same order as the input ranges.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if any individual read fails.
    pub async fn read_ranges(&self, req: BatchReadRequest<'_>) -> IoResult<Vec<RangeReadResult>> {
        if req.is_empty() {
            return Ok(Vec::new());
        }

        // Spawn one blocking task per request; collect handles preserving order.
        let handles: Vec<tokio::task::JoinHandle<IoResult<Arc<[u8]>>>> = req
            .paths
            .iter()
            .zip(req.ranges.iter())
            .map(|(path, range)| {
                let path_buf = path.to_path_buf();
                let offset = range.offset;
                let len = range.len;
                tokio::task::spawn_blocking(move || {
                    let bytes = load_range_blocking(&path_buf, offset, len)?;
                    Ok(bytes.into_shared())
                })
            })
            .collect();

        let mut results = Vec::with_capacity(handles.len());
        for (i, handle) in handles.into_iter().enumerate() {
            let bytes: Arc<[u8]> = handle
                .await
                .map_err(|_| std::io::Error::other("spawn_blocking panicked"))??;
            let logical_len = req.ranges[i].len;
            results.push(RangeReadResult {
                request_index: i,
                bytes,
                logical_offset: 0,
                logical_len,
            });
        }

        Ok(results)
    }

    /// Open `path` for async writing and return a [`TokioWriter`] bound to it.
    ///
    /// Creates any missing parent directories automatically.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created.
    pub async fn create_writer(&self, path: &Path) -> IoResult<TokioWriter> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = tokio::fs::File::create(path).await?;
        Ok(TokioWriter { file })
    }
}

// ============================================================================
// StorageEngine impl
// ============================================================================

impl super::StorageEngine for TokioStorage {
    fn kind(&self) -> StorageKind {
        StorageKind::Tokio
    }

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        StorageAvailability::Available
    }

    fn capabilities() -> StorageCapabilities
    where
        Self: Sized,
    {
        StorageCapabilities::probe()
    }
}

// ============================================================================
// TokioWriter
// ============================================================================

/// A file handle opened for async writes.
///
/// Obtain via [`TokioStorage::create_writer`].
pub struct TokioWriter {
    file: tokio::fs::File,
}

impl std::fmt::Debug for TokioWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokioWriter").finish_non_exhaustive()
    }
}

impl TokioWriter {
    /// Write `req.data` starting at `req.offset` within the open file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the write fails.
    pub async fn write_at(&mut self, req: WriteAtRequest<'_>) -> IoResult<()> {
        use tokio::io::{AsyncSeekExt, AsyncWriteExt};
        self.file.seek(std::io::SeekFrom::Start(req.offset)).await?;
        self.file.write_all(req.data).await
    }

    /// Flush buffered data and synchronise to durable storage.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the sync fails.
    pub async fn flush(&mut self) -> IoResult<()> {
        self.file.sync_all().await
    }
}

impl super::StorageEngine for TokioWriter {
    fn kind(&self) -> StorageKind {
        StorageKind::Tokio
    }

    fn availability() -> StorageAvailability
    where
        Self: Sized,
    {
        StorageAvailability::Available
    }

    fn capabilities() -> StorageCapabilities
    where
        Self: Sized,
    {
        StorageCapabilities::probe()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{BatchRange, BatchReadRequest, StorageEngine};
    use tempfile::TempDir;

    fn write_tmp(dir: &TempDir, name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn kind_is_tokio() {
        assert_eq!(TokioStorage::new().kind(), StorageKind::Tokio);
    }

    #[test]
    fn availability_is_available() {
        assert!(TokioStorage::availability().is_available());
    }

    #[tokio::test]
    async fn read_file_roundtrip() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let path = write_tmp(&dir, "file.bin", &data);

        let result = TokioStorage::new()
            .read_file(FileReadRequest::new(&path))
            .await
            .unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[tokio::test]
    async fn read_file_empty() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "empty.bin", b"");

        let result = TokioStorage::new()
            .read_file(FileReadRequest::new(&path))
            .await
            .unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn read_range_returns_correct_slice() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..100).collect();
        let path = write_tmp(&dir, "range.bin", &data);

        let result = TokioStorage::new()
            .read_range(RangeReadRequest::new(&path, 10, 20))
            .await
            .unwrap();
        assert_eq!(result.as_ref(), &data[10..30]);
    }

    #[tokio::test]
    async fn read_range_zero_len() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "z.bin", b"hello");

        let result = TokioStorage::new()
            .read_range(RangeReadRequest::new(&path, 0, 0))
            .await
            .unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn read_ranges_empty() {
        let results = TokioStorage::new()
            .read_ranges(BatchReadRequest::new(&[], &[]))
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn read_ranges_single() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..200).collect();
        let path = write_tmp(&dir, "batch.bin", &data);

        let paths = [path.as_path()];
        let ranges = [BatchRange::new(50, 30)];
        let results = TokioStorage::new()
            .read_ranges(BatchReadRequest::new(&paths, &ranges))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data(), &data[50..80]);
    }

    #[tokio::test]
    async fn read_ranges_multiple_preserves_order() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        let path = write_tmp(&dir, "multi.bin", &data);
        let p = path.as_path();

        let paths = [p, p, p];
        let ranges = [
            BatchRange::new(0, 10),
            BatchRange::new(20, 10),
            BatchRange::new(100, 5),
        ];
        let results = TokioStorage::new()
            .read_ranges(BatchReadRequest::new(&paths, &ranges))
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data(), &data[0..10]);
        assert_eq!(results[1].data(), &data[20..30]);
        assert_eq!(results[2].data(), &data[100..105]);
    }

    #[tokio::test]
    async fn write_at_and_flush_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("write.bin");

        let engine = TokioStorage::new();
        let mut writer = engine.create_writer(&path).await.unwrap();
        writer
            .write_at(WriteAtRequest::new(0, b"hello async"))
            .await
            .unwrap();
        writer.flush().await.unwrap();
        drop(writer);

        assert_eq!(std::fs::read(&path).unwrap(), b"hello async");
    }

    #[tokio::test]
    async fn writer_kind_is_tokio() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("w.bin");
        let engine = TokioStorage::new();
        let writer = engine.create_writer(&path).await.unwrap();
        assert_eq!(writer.kind(), StorageKind::Tokio);
    }
}
