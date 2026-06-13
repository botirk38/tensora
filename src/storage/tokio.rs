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

use crate::backends::{AsyncReader as BackendReader, AsyncWriter as BackendWriter};
use crate::storage::{
    BatchReadRequest, FileReadRequest, IoResult, RangeReadRequest, RangeReadResult, WriteAtRequest,
    availability::{StorageAvailability, StorageCapabilities, StorageKind},
    buffer::OwnedBytes,
};

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
        let mut reader = BackendReader::new();
        convert_bytes(reader.load(req.path).await?)
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
        let mut reader = BackendReader::new();
        convert_bytes(reader.load_range(req.path, req.offset, req.len).await?)
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

        // Build the backend BatchRequest slice: (PathBuf, offset, len)
        let backend_requests: Vec<crate::backends::BatchRequest> = req
            .paths
            .iter()
            .zip(req.ranges.iter())
            .map(|(path, range)| (path.to_path_buf(), range.offset, range.len))
            .collect();

        let mut reader = BackendReader::new();
        let flattened: Vec<crate::backends::batch::FlattenedResult> =
            reader.load_range_batch(&backend_requests).await?;

        // `flattened` is Vec<FlattenedResult> = Vec<(Arc<[u8]>, usize, usize)>
        // where each entry corresponds 1:1 to the original request order.
        let results = flattened
            .into_iter()
            .enumerate()
            .map(|(i, (bytes, logical_offset, logical_len))| RangeReadResult {
                request_index: i,
                bytes,
                logical_offset,
                logical_len,
            })
            .collect();

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
        Ok(TokioWriter { inner: BackendWriter::create(path).await? })
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
    inner: BackendWriter,
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
        self.inner.write_at(req.offset, req.data).await
    }

    /// Flush buffered data and synchronise to durable storage.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the sync fails.
    pub async fn flush(&mut self) -> IoResult<()> {
        self.inner.sync_all().await
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
// Helper: convert backends::byte::OwnedBytes → storage::buffer::OwnedBytes
// ============================================================================

fn convert_bytes(b: crate::backends::byte::OwnedBytes) -> IoResult<OwnedBytes> {
    use crate::backends::byte::OwnedBytes as B;
    Ok(match b {
        B::Pooled(p) => OwnedBytes::Pooled(p),
        #[cfg(target_os = "linux")]
        B::Aligned(a) => OwnedBytes::Aligned(a),
        B::Shared(s) => OwnedBytes::Shared(s),
        B::Mmap(m) => OwnedBytes::Mmap(m),
        B::Vec(v) => OwnedBytes::Vec(v),
    })
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

        let result = TokioStorage::new().read_file(FileReadRequest::new(&path)).await.unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[tokio::test]
    async fn read_file_empty() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "empty.bin", b"");

        let result = TokioStorage::new().read_file(FileReadRequest::new(&path)).await.unwrap();
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
        let ranges = [BatchRange::new(0, 10), BatchRange::new(20, 10), BatchRange::new(100, 5)];
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
        writer.write_at(WriteAtRequest::new(0, b"hello async")).await.unwrap();
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
