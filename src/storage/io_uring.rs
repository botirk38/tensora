//! io_uring storage engine (Linux only).
//!
//! [`IoUringStorage`] implements [`ReadableStorage`] and [`StorageEngine`].
//! It uses the kernel's io_uring interface for high-throughput, low-latency
//! batch reads via a persistent submission/completion ring, with optional
//! SQ polling and O_DIRECT support.
//!
//! Write access is obtained by calling [`IoUringStorage::create_writer`],
//! which returns an [`IoUringWriter`] bound to a specific file path.
//!
//! io_uring is only available on Linux; this module is entirely
//! `#[cfg(target_os = "linux")]`.

#[cfg(target_os = "linux")]
mod inner {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use crate::backends::io_uring::{Reader as BackendReader, Writer as BackendWriter};
    use crate::storage::{
        BatchReadRequest, FileReadRequest, IoResult, RangeReadRequest, RangeReadResult,
        WriteAtRequest,
        availability::{StorageAvailability, StorageCapabilities, StorageKind, UnavailableReason},
        buffer::OwnedBytes,
    };

    // ============================================================================
    // IoUringStorage
    // ============================================================================

    /// High-throughput io_uring storage engine (Linux only).
    ///
    /// Implements [`ReadableStorage`] and [`StorageEngine`].  Creates a fresh
    /// `io_uring` ring per operation; for sustained throughput construct a
    /// single engine and reuse it across many calls.
    ///
    /// ```rust,ignore
    /// use tensora::storage::io_uring::IoUringStorage;
    /// use tensora::storage::{FileReadRequest, ReadableStorage};
    /// use std::path::Path;
    ///
    /// let engine = IoUringStorage::new()?;
    /// let bytes = engine.read_file(FileReadRequest::new(Path::new("model.bin")))?;
    /// ```
    ///
    /// [`ReadableStorage`]: crate::storage::ReadableStorage
    #[derive(Debug, Clone, Copy, Default)]
    pub struct IoUringStorage;

    impl IoUringStorage {
        /// Create a new `IoUringStorage` engine.
        ///
        /// Does not probe availability — check [`IoUringStorage::availability`]
        /// before constructing if you need to handle the unavailable case.
        #[inline]
        #[must_use]
        pub const fn new() -> Self {
            Self
        }

        /// Open `path` for writing and return an [`IoUringWriter`] bound to it.
        ///
        /// Creates any missing parent directories automatically.
        ///
        /// # Errors
        ///
        /// Returns an I/O error if io_uring is unavailable, or if the file
        /// cannot be created.
        pub fn create_writer(&self, path: &Path) -> IoResult<IoUringWriter> {
            let _writer = BackendWriter::create(path)?;
            Ok(IoUringWriter { path: path.to_path_buf() })
        }
    }

    // ============================================================================
    // StorageEngine impl
    // ============================================================================

    impl super::super::StorageEngine for IoUringStorage {
        fn kind(&self) -> StorageKind {
            StorageKind::IoUring
        }

        fn availability() -> StorageAvailability
        where
            Self: Sized,
        {
            match crate::backends::io_uring::availability() {
                crate::backends::availability::BackendAvailability::Available => {
                    StorageAvailability::Available
                }
                crate::backends::availability::BackendAvailability::Unavailable {
                    reason,
                    details,
                } => {
                    use crate::backends::availability::BackendUnavailableReason;
                    let r = match reason {
                        BackendUnavailableReason::UnsupportedPlatform => {
                            UnavailableReason::UnsupportedPlatform
                        }
                        BackendUnavailableReason::PermissionDenied => {
                            UnavailableReason::PermissionDenied
                        }
                        BackendUnavailableReason::MissingKernelFeature => {
                            UnavailableReason::MissingKernelFeature
                        }
                        BackendUnavailableReason::InvalidKernelConfiguration => {
                            UnavailableReason::InvalidKernelConfiguration
                        }
                        BackendUnavailableReason::MissingDependency => {
                            UnavailableReason::MissingDependency
                        }
                        BackendUnavailableReason::FilesystemUnsupported => {
                            UnavailableReason::FilesystemUnsupported
                        }
                        BackendUnavailableReason::Other(msg) => UnavailableReason::Other(msg),
                    };
                    StorageAvailability::unavailable(r, details)
                }
            }
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

    impl super::super::ReadableStorage for IoUringStorage {
        fn read_file(&self, req: FileReadRequest<'_>) -> IoResult<OwnedBytes> {
            let mut reader = BackendReader::new();
            convert_bytes(reader.load(req.path)?)
        }

        fn read_range(&self, req: RangeReadRequest<'_>) -> IoResult<OwnedBytes> {
            let mut reader = BackendReader::new();
            convert_bytes(reader.load_range(req.path, req.offset, req.len)?)
        }

        fn read_ranges(&self, req: BatchReadRequest<'_>) -> IoResult<Vec<RangeReadResult>> {
            if req.is_empty() {
                return Ok(Vec::new());
            }

            let backend_requests: Vec<crate::backends::BatchRequest> = req
                .paths
                .iter()
                .zip(req.ranges.iter())
                .map(|(path, range)| (path.to_path_buf(), range.offset, range.len))
                .collect();

            let mut reader = BackendReader::new();
            let flattened: Vec<crate::backends::batch::FlattenedResult> =
                reader.load_range_batch(&backend_requests)?;

            Ok(flattened
                .into_iter()
                .enumerate()
                .map(|(i, (bytes, logical_offset, logical_len))| RangeReadResult {
                    request_index: i,
                    bytes,
                    logical_offset,
                    logical_len,
                })
                .collect())
        }
    }

    // ============================================================================
    // IoUringWriter
    // ============================================================================

    /// A file opened for io_uring writes.
    ///
    /// Unlike `SyncWriter` / `TokioWriter` which hold an open file descriptor,
    /// `IoUringWriter` reopens the file per [`write_at`] call — the io_uring
    /// backend manages file descriptors internally within each operation.
    ///
    /// Obtain via [`IoUringStorage::create_writer`].
    ///
    /// [`write_at`]: IoUringWriter::write_at
    #[derive(Debug)]
    pub struct IoUringWriter {
        path: PathBuf,
    }

    impl super::super::StorageEngine for IoUringWriter {
        fn kind(&self) -> StorageKind {
            StorageKind::IoUring
        }

        fn availability() -> StorageAvailability
        where
            Self: Sized,
        {
            IoUringStorage::availability()
        }

        fn capabilities() -> StorageCapabilities
        where
            Self: Sized,
        {
            StorageCapabilities::probe()
        }
    }

    impl super::super::WritableStorage for IoUringWriter {
        fn write_at(&mut self, req: WriteAtRequest<'_>) -> IoResult<()> {
            let mut writer = BackendWriter::new()?;
            writer.write_at(&self.path, req.offset, req.data)
        }

        fn flush(&mut self) -> IoResult<()> {
            Ok(())
        }
    }

    // ============================================================================
    // Helper: convert backends::byte::OwnedBytes → storage::buffer::OwnedBytes
    // ============================================================================

    fn convert_bytes(b: crate::backends::byte::OwnedBytes) -> IoResult<OwnedBytes> {
        use crate::backends::byte::OwnedBytes as B;
        Ok(match b {
            B::Pooled(p) => OwnedBytes::Pooled(p),
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
        use crate::storage::{
            BatchRange, BatchReadRequest, MappableStorage, ReadableStorage, StorageEngine,
            WritableStorage,
        };
        use tempfile::TempDir;

        fn write_tmp(dir: &TempDir, name: &str, data: &[u8]) -> std::path::PathBuf {
            let path = dir.path().join(name);
            std::fs::write(&path, data).unwrap();
            path
        }

        fn skip_if_unavailable() -> bool {
            !IoUringStorage::availability().is_available()
        }

        #[test]
        fn kind_is_io_uring() {
            assert_eq!(IoUringStorage::new().kind(), StorageKind::IoUring);
        }

        #[test]
        fn availability_reflects_kernel() {
            // Just assert it returns without panic — value depends on the host.
            let _ = IoUringStorage::availability();
        }

        #[test]
        fn read_file_roundtrip() {
            if skip_if_unavailable() {
                return;
            }
            let dir = TempDir::new().unwrap();
            let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
            let path = write_tmp(&dir, "file.bin", &data);

            let result =
                IoUringStorage::new().read_file(FileReadRequest::new(&path)).unwrap();
            assert_eq!(result.as_ref(), &data[..]);
        }

        #[test]
        fn read_file_empty() {
            if skip_if_unavailable() {
                return;
            }
            let dir = TempDir::new().unwrap();
            let path = write_tmp(&dir, "empty.bin", b"");

            let result =
                IoUringStorage::new().read_file(FileReadRequest::new(&path)).unwrap();
            assert!(result.is_empty());
        }

        #[test]
        fn read_range_returns_correct_slice() {
            if skip_if_unavailable() {
                return;
            }
            let dir = TempDir::new().unwrap();
            let data: Vec<u8> = (0u8..100).collect();
            let path = write_tmp(&dir, "range.bin", &data);

            let result = IoUringStorage::new()
                .read_range(RangeReadRequest::new(&path, 10, 20))
                .unwrap();
            assert_eq!(result.as_ref(), &data[10..30]);
        }

        #[test]
        fn read_range_zero_len() {
            if skip_if_unavailable() {
                return;
            }
            let dir = TempDir::new().unwrap();
            let path = write_tmp(&dir, "z.bin", b"hello");

            let result = IoUringStorage::new()
                .read_range(RangeReadRequest::new(&path, 0, 0))
                .unwrap();
            assert!(result.is_empty());
        }

        #[test]
        fn read_ranges_empty() {
            if skip_if_unavailable() {
                return;
            }
            let results = IoUringStorage::new()
                .read_ranges(BatchReadRequest::new(&[], &[]))
                .unwrap();
            assert!(results.is_empty());
        }

        #[test]
        fn read_ranges_single() {
            if skip_if_unavailable() {
                return;
            }
            let dir = TempDir::new().unwrap();
            let data: Vec<u8> = (0u8..200).collect();
            let path = write_tmp(&dir, "batch.bin", &data);

            let paths = [path.as_path()];
            let ranges = [BatchRange::new(50, 30)];
            let results = IoUringStorage::new()
                .read_ranges(BatchReadRequest::new(&paths, &ranges))
                .unwrap();

            assert_eq!(results.len(), 1);
            assert_eq!(results[0].data(), &data[50..80]);
        }

        #[test]
        fn read_ranges_multiple_preserves_order() {
            if skip_if_unavailable() {
                return;
            }
            let dir = TempDir::new().unwrap();
            let data: Vec<u8> = (0u8..=255).collect();
            let path = write_tmp(&dir, "multi.bin", &data);
            let p = path.as_path();

            let paths = [p, p, p];
            let ranges =
                [BatchRange::new(0, 10), BatchRange::new(20, 10), BatchRange::new(100, 5)];
            let results = IoUringStorage::new()
                .read_ranges(BatchReadRequest::new(&paths, &ranges))
                .unwrap();

            assert_eq!(results.len(), 3);
            assert_eq!(results[0].data(), &data[0..10]);
            assert_eq!(results[1].data(), &data[20..30]);
            assert_eq!(results[2].data(), &data[100..105]);
        }

        #[test]
        fn write_at_and_flush_roundtrip() {
            if skip_if_unavailable() {
                return;
            }
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("write.bin");

            let mut writer = IoUringStorage::new().create_writer(&path).unwrap();
            writer.write_at(WriteAtRequest::new(0, b"hello io_uring")).unwrap();
            writer.flush().unwrap();
            drop(writer);

            assert_eq!(std::fs::read(&path).unwrap(), b"hello io_uring");
        }

        #[test]
        fn writer_kind_is_io_uring() {
            if skip_if_unavailable() {
                return;
            }
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("w.bin");
            let writer = IoUringStorage::new().create_writer(&path).unwrap();
            assert_eq!(writer.kind(), StorageKind::IoUring);
        }
    }
}

// ============================================================================
// Public re-exports (Linux only)
// ============================================================================

#[cfg(target_os = "linux")]
pub use inner::{IoUringStorage, IoUringWriter};
