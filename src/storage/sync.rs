//! Synchronous blocking storage engine.
//!
//! [`SyncStorage`] implements [`ReadableStorage`] and [`StorageEngine`].
//! On Linux it defaults to O_DIRECT where possible; on other platforms it
//! uses buffered `std::fs` I/O.
//!
//! Write access is obtained by calling [`SyncStorage::create_writer`], which
//! returns a [`SyncWriter`] that holds an open file handle and implements
//! [`WritableStorage`].

use std::path::Path;

use crate::backends::{SyncReader as BackendReader, SyncWriter as BackendWriter};
use crate::storage::{
    FileReadRequest, IoResult, RangeReadRequest, WriteAtRequest,
    availability::{StorageAvailability, StorageCapabilities, StorageKind},
    buffer::OwnedBytes,
};

// ============================================================================
// SyncStorage
// ============================================================================

/// Synchronous blocking storage engine.
///
/// ```rust,ignore
/// use tensora::storage::sync::SyncStorage;
/// use tensora::storage::{FileReadRequest, ReadableStorage};
///
/// let engine = SyncStorage::default();
/// let bytes = engine.read_file(FileReadRequest::new(Path::new("model.safetensors")))?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SyncStorage;

impl Default for SyncStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl SyncStorage {
    /// Create a new `SyncStorage` engine.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Open `path` for writing and return a [`SyncWriter`] bound to it.
    ///
    /// Creates any missing parent directories automatically.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created.
    pub fn create_writer(&self, path: &Path) -> IoResult<SyncWriter> {
        Ok(SyncWriter { inner: BackendWriter::create(path)? })
    }
}

// ============================================================================
// StorageEngine impl
// ============================================================================

impl super::StorageEngine for SyncStorage {
    fn kind(&self) -> StorageKind {
        StorageKind::Sync
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
// ReadableStorage impl
// ============================================================================

impl super::ReadableStorage for SyncStorage {
    fn read_file(&self, req: FileReadRequest<'_>) -> IoResult<OwnedBytes> {
        let mut reader = BackendReader::new();
        Ok(convert_bytes(reader.load(req.path)?))
    }

    fn read_range(&self, req: RangeReadRequest<'_>) -> IoResult<OwnedBytes> {
        let mut reader = BackendReader::new();
        Ok(convert_bytes(reader.load_range(req.path, req.offset, req.len)?))
    }
}

// ============================================================================
// SyncWriter + WritableStorage
// ============================================================================

/// A file handle opened for synchronous writes.
///
/// Obtain via [`SyncStorage::create_writer`].
pub struct SyncWriter {
    inner: BackendWriter,
}

impl std::fmt::Debug for SyncWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncWriter").finish_non_exhaustive()
    }
}

impl super::StorageEngine for SyncWriter {
    fn kind(&self) -> StorageKind {
        StorageKind::Sync
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

impl super::WritableStorage for SyncWriter {
    fn write_at(&mut self, req: WriteAtRequest<'_>) -> IoResult<()> {
        self.inner.write_at(req.offset, req.data)
    }

    fn flush(&mut self) -> IoResult<()> {
        self.inner.sync_all()
    }
}

// ============================================================================
// Helper: convert backends::byte::OwnedBytes → storage::buffer::OwnedBytes
// ============================================================================

fn convert_bytes(b: crate::backends::byte::OwnedBytes) -> OwnedBytes {
    use crate::backends::byte::OwnedBytes as B;
    match b {
        B::Pooled(p) => OwnedBytes::Pooled(p),
        #[cfg(target_os = "linux")]
        B::Aligned(a) => OwnedBytes::Aligned(a),
        B::Shared(s) => OwnedBytes::Shared(s),
        B::Mmap(m) => OwnedBytes::Mmap(m),
        B::Vec(v) => OwnedBytes::Vec(v),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{BatchRange, BatchReadRequest, ReadableStorage, StorageEngine, WritableStorage};
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

        let result = SyncStorage::default().read_file(FileReadRequest::new(&path)).unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[test]
    fn read_file_empty() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "empty.bin", b"");

        let result = SyncStorage::default().read_file(FileReadRequest::new(&path)).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_range_returns_correct_slice() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..100).collect();
        let path = write_tmp(&dir, "range.bin", &data);

        let result =
            SyncStorage::default().read_range(RangeReadRequest::new(&path, 10, 20)).unwrap();
        assert_eq!(result.as_ref(), &data[10..30]);
    }

    #[test]
    fn read_range_zero_len() {
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "z.bin", b"hello");

        let result =
            SyncStorage::default().read_range(RangeReadRequest::new(&path, 0, 0)).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_ranges_empty() {
        let results =
            SyncStorage::default().read_ranges(BatchReadRequest::new(&[], &[])).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn read_ranges_single() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..200).collect();
        let path = write_tmp(&dir, "batch.bin", &data);

        let paths = [path.as_path()];
        let ranges = [BatchRange::new(50, 30)];
        let results =
            SyncStorage::default().read_ranges(BatchReadRequest::new(&paths, &ranges)).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data(), &data[50..80]);
    }

    #[test]
    fn read_ranges_multiple_preserves_order() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        let path = write_tmp(&dir, "multi.bin", &data);
        let p = path.as_path();

        let paths = [p, p, p];
        let ranges = [BatchRange::new(0, 10), BatchRange::new(20, 10), BatchRange::new(100, 5)];
        let results =
            SyncStorage::default().read_ranges(BatchReadRequest::new(&paths, &ranges)).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data(), &data[0..10]);
        assert_eq!(results[1].data(), &data[20..30]);
        assert_eq!(results[2].data(), &data[100..105]);
    }

    #[test]
    fn read_ranges_adjacent_correct() {
        // Two adjacent ranges on the same file are read independently.
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        let path = write_tmp(&dir, "adjacent.bin", &data);
        let p = path.as_path();

        let paths = [p, p];
        let ranges = [BatchRange::new(0, 50), BatchRange::new(50, 50)];
        let results =
            SyncStorage::default().read_ranges(BatchReadRequest::new(&paths, &ranges)).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data(), &data[0..50]);
        assert_eq!(results[1].data(), &data[50..100]);
    }

    #[test]
    fn write_at_and_flush_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("write.bin");

        let mut writer = SyncStorage::default().create_writer(&path).unwrap();
        writer.write_at(WriteAtRequest::new(0, b"hello world")).unwrap();
        writer.flush().unwrap();
        drop(writer);

        assert_eq!(std::fs::read(&path).unwrap(), b"hello world");
    }

    #[test]
    fn kind_is_sync() {
        assert_eq!(SyncStorage::default().kind(), StorageKind::Sync);
    }

    #[test]
    fn availability_is_available() {
        assert!(SyncStorage::availability().is_available());
    }
}
