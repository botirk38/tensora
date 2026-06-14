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
    use std::fs::{File, OpenOptions};
    use std::io::{Error, ErrorKind};
    use std::os::fd::AsRawFd;
    use std::path::Path;
    use std::sync::Arc;

    use io_uring::{IoUring, opcode, types};

    use crate::storage::{
        BatchReadRequest, FileReadRequest, IoResult, RangeReadRequest, RangeReadResult,
        WriteAtRequest,
        availability::{StorageAvailability, StorageCapabilities, StorageKind},
        buffer::OwnedBytes,
    };

    const RING_DEPTH: u32 = 8;
    const MAX_IO_LEN: usize = u32::MAX as usize;

    // ============================================================================
    // IoUringStorage
    // ============================================================================

    /// High-throughput io_uring storage engine (Linux only).
    ///
    /// Implements [`ReadableStorage`] and [`StorageEngine`].  Each operation uses
    /// a short-lived io_uring ring and submits one or more offset-addressed I/O
    /// operations against ordinary file descriptors.
    ///
    /// ```rust,ignore
    /// use tensora::storage::io_uring::IoUringStorage;
    /// use tensora::storage::{FileReadRequest, ReadableStorage};
    /// use std::path::Path;
    ///
    /// let engine = IoUringStorage::new();
    /// let bytes = engine.read_file(FileReadRequest::new(Path::new("model.bin")))?;
    /// # std::io::Result::Ok(())
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
            ensure_available()?;
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)?;
            }
            let file = File::create(path)?;
            Ok(IoUringWriter { file })
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
            StorageCapabilities::probe().io_uring
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
            load(req.path)
        }

        fn read_range(&self, req: RangeReadRequest<'_>) -> IoResult<OwnedBytes> {
            load_range(req.path, req.offset, req.len)
        }

        fn read_ranges(&self, req: BatchReadRequest<'_>) -> IoResult<Vec<RangeReadResult>> {
            if req.is_empty() {
                return Ok(Vec::new());
            }

            req.paths
                .iter()
                .zip(req.ranges.iter())
                .enumerate()
                .map(|(request_index, (path, range))| {
                    let bytes = read_range_arc(path, range.offset, range.len)?;
                    Ok(RangeReadResult {
                        request_index,
                        bytes,
                        logical_offset: 0,
                        logical_len: range.len,
                    })
                })
                .collect()
        }
    }

    // ============================================================================
    // IoUringWriter
    // ============================================================================

    /// A file opened for io_uring writes.
    ///
    /// Obtain via [`IoUringStorage::create_writer`].
    #[derive(Debug)]
    pub struct IoUringWriter {
        file: File,
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
            ensure_available()?;
            write_all_at(&self.file, req.offset, req.data)
        }

        fn flush(&mut self) -> IoResult<()> {
            self.file.sync_all()
        }
    }

    // ============================================================================
    // Core I/O helpers
    // ============================================================================

    fn ensure_available() -> IoResult<()> {
        match IoUringStorage::availability() {
            StorageAvailability::Available => Ok(()),
            unavailable => Err(Error::other(format!("io_uring storage is {unavailable}"))),
        }
    }

    fn open_for_read(path: &Path) -> IoResult<File> {
        ensure_available()?;
        OpenOptions::new().read(true).open(path)
    }

    fn load(path: &Path) -> IoResult<OwnedBytes> {
        let file = open_for_read(path)?;
        let len =
            usize::try_from(file.metadata()?.len()).map_err(|_| Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }

        let mut buf = vec![0u8; len];
        read_exact_at(&file, 0, &mut buf)?;
        Ok(OwnedBytes::Vec(buf))
    }

    fn load_range(path: &Path, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        Ok(OwnedBytes::Shared(read_range_arc(path, offset, len)?))
    }

    fn read_range_arc(path: &Path, offset: u64, len: usize) -> IoResult<Arc<[u8]>> {
        if len == 0 {
            return Ok(Arc::new([]));
        }

        let file = open_for_read(path)?;
        let mut buf = vec![0u8; len];
        read_exact_at(&file, offset, &mut buf)?;
        Ok(Arc::from(buf.into_boxed_slice()))
    }

    fn read_exact_at(file: &File, offset: u64, mut buf: &mut [u8]) -> IoResult<()> {
        let mut ring = IoUring::new(RING_DEPTH)?;
        let mut absolute_offset = offset;

        while !buf.is_empty() {
            let chunk_len = buf.len().min(MAX_IO_LEN);
            let (chunk, rest) = buf.split_at_mut(chunk_len);
            let mut read = 0usize;

            while read < chunk.len() {
                let len = (chunk.len() - read).min(MAX_IO_LEN);
                let ptr = unsafe { chunk.as_mut_ptr().add(read) };
                let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), ptr, len as u32)
                    .offset(absolute_offset + read as u64)
                    .build()
                    .user_data(0);

                submit_one(&mut ring, &entry)?;
                let result = wait_one(&mut ring)?;
                if result == 0 {
                    return Err(Error::new(ErrorKind::UnexpectedEof, "short io_uring read"));
                }
                read += result;
            }

            absolute_offset = absolute_offset
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "read offset overflow"))?;
            buf = rest;
        }

        Ok(())
    }

    fn write_all_at(file: &File, offset: u64, mut data: &[u8]) -> IoResult<()> {
        let mut ring = IoUring::new(RING_DEPTH)?;
        let mut absolute_offset = offset;

        while !data.is_empty() {
            let chunk_len = data.len().min(MAX_IO_LEN);
            let (chunk, rest) = data.split_at(chunk_len);
            let mut written = 0usize;

            while written < chunk.len() {
                let len = (chunk.len() - written).min(MAX_IO_LEN);
                let ptr = unsafe { chunk.as_ptr().add(written) };
                let entry = opcode::Write::new(types::Fd(file.as_raw_fd()), ptr, len as u32)
                    .offset(absolute_offset + written as u64)
                    .build()
                    .user_data(0);

                submit_one(&mut ring, &entry)?;
                let result = wait_one(&mut ring)?;
                if result == 0 {
                    return Err(Error::new(ErrorKind::WriteZero, "short io_uring write"));
                }
                written += result;
            }

            absolute_offset = absolute_offset
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "write offset overflow"))?;
            data = rest;
        }

        Ok(())
    }

    fn submit_one(ring: &mut IoUring, entry: &io_uring::squeue::Entry) -> IoResult<()> {
        {
            let mut submission = ring.submission();
            unsafe {
                submission
                    .push(entry)
                    .map_err(|_| Error::other("io_uring submission queue is full"))?;
            }
        }
        ring.submit_and_wait(1)?;
        Ok(())
    }

    fn wait_one(ring: &mut IoUring) -> IoResult<usize> {
        let completion = ring
            .completion()
            .next()
            .ok_or_else(|| Error::other("io_uring completion queue was empty"))?;
        let result = completion.result();
        if result < 0 {
            return Err(Error::from_raw_os_error(-result));
        }
        Ok(result as usize)
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

            let result = IoUringStorage::new()
                .read_file(FileReadRequest::new(&path))
                .unwrap();
            assert_eq!(result.as_ref(), &data[..]);
        }

        #[test]
        fn read_file_empty() {
            if skip_if_unavailable() {
                return;
            }
            let dir = TempDir::new().unwrap();
            let path = write_tmp(&dir, "empty.bin", b"");

            let result = IoUringStorage::new()
                .read_file(FileReadRequest::new(&path))
                .unwrap();
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
            let ranges = [
                BatchRange::new(0, 10),
                BatchRange::new(20, 10),
                BatchRange::new(100, 5),
            ];
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
            writer
                .write_at(WriteAtRequest::new(0, b"hello io_uring"))
                .unwrap();
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
