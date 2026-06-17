//! Linux io_uring storage implementation with batched I/O.
//!
//! Single-range and single-slice operations use a dedicated ring per call.
//! Batch operations (`read_ranges`, `write_positioned_file`, `write_slices`)
//! saturate a shared ring: up to `ring_depth` ops are in-flight at once,
//! completions are drained in a tight loop, and partial results are
//! immediately resubmitted.

use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;

use io_uring::{IoUring as Uring, opcode, types};

use crate::io::{
    ByteRange, FileRange, Io, IoResult, RangeRead, WriteSlice, WriteSlices,
    availability::{IoAvailability, IoKind},
    buffer::OwnedBytes,
};

const DEFAULT_RING_DEPTH: u32 = 256;
/// Maximum single-op transfer size (io_uring length field is u32).
const MAX_IO_LEN: usize = u32::MAX as usize;

// ============================================================================
// IoUring
// ============================================================================

/// High-throughput io_uring I/O backend (Linux only).
#[derive(Debug, Clone, Copy, Default)]
pub struct IoUring {
    options: IoUringOptions,
}

/// Options for the io_uring I/O backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoUringOptions {
    /// Depth of each io_uring instance. Also caps batch in-flight operations.
    pub ring_depth: u32,
}

impl Default for IoUringOptions {
    fn default() -> Self {
        Self {
            ring_depth: DEFAULT_RING_DEPTH,
        }
    }
}

impl IoUring {
    /// Create a new `IoUring` backend with default options.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            options: IoUringOptions {
                ring_depth: DEFAULT_RING_DEPTH,
            },
        }
    }

    pub fn with_options(options: IoUringOptions) -> IoResult<Self> {
        if options.ring_depth == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "ring_depth must be greater than zero",
            ));
        }
        Ok(Self { options })
    }

    #[inline]
    #[must_use]
    pub const fn options(&self) -> &IoUringOptions {
        &self.options
    }

    fn ensure_available() -> IoResult<()> {
        match IoUring::availability() {
            IoAvailability::Available => Ok(()),
            unavailable => Err(Error::other(format!("io_uring storage is {unavailable}"))),
        }
    }

    fn open_create_truncate(path: &Path) -> IoResult<File> {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
    }

    fn open_write_existing(path: &Path) -> IoResult<File> {
        OpenOptions::new().write(true).open(path)
    }

    fn read_exact_at(&self, file: &File, offset: u64, mut buf: &mut [u8]) -> IoResult<()> {
        let mut ring = Uring::new(self.options.ring_depth)?;
        let mut absolute_offset = offset;

        while !buf.is_empty() {
            let chunk_len = buf.len().min(MAX_IO_LEN);
            let (chunk, rest) = buf.split_at_mut(chunk_len);
            let mut read = 0usize;

            while read < chunk.len() {
                let len = (chunk.len() - read).min(MAX_IO_LEN);
                // SAFETY: `read < chunk.len()` and `len` is bounded by the
                // remaining bytes, so the resulting pointer stays within
                // `chunk` for the submitted read.
                let ptr = unsafe { chunk.as_mut_ptr().add(read) };
                let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), ptr, len as u32)
                    .offset(absolute_offset + read as u64)
                    .build()
                    .user_data(0);

                Self::submit_one(&mut ring, &entry)?;
                let result = Self::wait_one(&mut ring)?;
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

    fn write_exact_at(&self, file: &File, offset: u64, mut data: &[u8]) -> IoResult<()> {
        let mut ring = Uring::new(self.options.ring_depth)?;
        let mut absolute_offset = offset;

        while !data.is_empty() {
            let chunk_len = data.len().min(MAX_IO_LEN);
            let (chunk, rest) = data.split_at(chunk_len);
            let mut written = 0usize;

            while written < chunk.len() {
                let len = (chunk.len() - written).min(MAX_IO_LEN);
                // SAFETY: `written < chunk.len()` and `len` is bounded by the
                // remaining bytes, so the resulting pointer stays within
                // `chunk` for the submitted write.
                let ptr = unsafe { chunk.as_ptr().add(written) };
                let entry = opcode::Write::new(types::Fd(file.as_raw_fd()), ptr, len as u32)
                    .offset(absolute_offset + written as u64)
                    .build()
                    .user_data(0);

                Self::submit_one(&mut ring, &entry)?;
                let result = Self::wait_one(&mut ring)?;
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

    /// Writes every slice in `writes` to `file` using a saturated io_uring ring.
    ///
    /// Submits up to `ring_depth` ops simultaneously.  Partial writes are
    /// resubmitted immediately.  `writes` must be non-overlapping (guaranteed
    /// by [`WriteSlices`]).
    fn ring_batch_writes(&self, file: &File, writes: &[WriteSlice<'_>]) -> IoResult<()> {
        if writes.is_empty() {
            return Ok(());
        }
        let n = writes.len();
        let mut done = vec![0usize; n];
        let mut ring = Uring::new(self.options.ring_depth)?;
        let mut in_flight: u32 = 0;
        let mut next_submit = 0usize;

        loop {
            while in_flight < self.options.ring_depth && next_submit < n {
                let idx = next_submit;
                let w = &writes[idx];
                let so_far = done[idx];
                let remaining = w.data.len() - so_far;
                if remaining == 0 {
                    next_submit += 1;
                    continue;
                }
                let len = remaining.min(MAX_IO_LEN) as u32;
                // SAFETY: `w.data` is a shared slice with lifetime tied to the
                // caller's `WriteSlice`.  The ring holds a reference to this
                // memory until the corresponding CQE is consumed below —
                // `writes` (and therefore `w.data`) outlives the ring in this
                // stack frame.
                let ptr = unsafe { w.data.as_ptr().add(so_far) };
                let entry = opcode::Write::new(types::Fd(file.as_raw_fd()), ptr, len)
                    .offset(w.offset + so_far as u64)
                    .build()
                    .user_data(idx as u64);
                {
                    let mut sq = ring.submission();
                    // SAFETY: see pointer comment above.
                    unsafe {
                        sq.push(&entry)
                            .map_err(|_| Error::other("io_uring SQ full"))?;
                    }
                }
                in_flight += 1;
                next_submit += 1;
            }

            if in_flight == 0 {
                break;
            }

            ring.submit_and_wait(1)?;

            let cq: Vec<_> = ring.completion().collect();
            for cqe in cq {
                in_flight -= 1;
                let idx = cqe.user_data() as usize;
                let result = cqe.result();
                if result < 0 {
                    return Err(Error::from_raw_os_error(-result));
                }
                let n_written = result as usize;
                if n_written == 0 {
                    return Err(Error::new(ErrorKind::WriteZero, "short io_uring write"));
                }
                done[idx] += n_written;
                if done[idx] < writes[idx].data.len() {
                    if next_submit > idx {
                        next_submit = idx;
                    }
                }
            }
        }

        Ok(())
    }

    fn submit_one(ring: &mut Uring, entry: &io_uring::squeue::Entry) -> IoResult<()> {
        {
            let mut submission = ring.submission();
            // SAFETY: `entry` references buffers owned by the caller and those
            // buffers remain alive until `submit_and_wait` completes below.
            unsafe {
                submission
                    .push(entry)
                    .map_err(|_| Error::other("io_uring submission queue is full"))?;
            }
        }
        ring.submit_and_wait(1)?;
        Ok(())
    }

    fn wait_one(ring: &mut Uring) -> IoResult<usize> {
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
}

impl super::Io for IoUring {
    const KIND: IoKind = IoKind::IoUring;

    fn availability() -> IoAvailability
    where
        Self: Sized,
    {
        crate::io::availability::IoCapabilities::cached()
            .io_uring
            .clone()
    }
}

impl super::BlockingIo for IoUring {
    fn read_file(&self, path: &Path) -> IoResult<OwnedBytes> {
        Self::ensure_available()?;
        let file = OpenOptions::new().read(true).open(path)?;
        let len =
            usize::try_from(file.metadata()?.len()).map_err(|_| Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut buf = vec![0u8; len];
        self.read_exact_at(&file, 0, &mut buf)?;
        Ok(OwnedBytes::Vec(buf))
    }

    fn read_range(&self, path: &Path, range: ByteRange) -> IoResult<OwnedBytes> {
        if range.is_empty() {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        Self::ensure_available()?;
        let file = OpenOptions::new().read(true).open(path)?;
        let mut buf = vec![0u8; range.len_usize()?];
        self.read_exact_at(&file, range.start(), &mut buf)?;
        Ok(OwnedBytes::Vec(buf))
    }

    fn read_ranges(&self, ranges: &[FileRange<'_>]) -> IoResult<Vec<RangeRead>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        Self::ensure_available()?;

        // Allocate a buffer for each range and open each file.
        let n = ranges.len();
        let mut bufs: Vec<Vec<u8>> = ranges
            .iter()
            .map(|e| e.range.len_usize().map(|len| vec![0u8; len]))
            .collect::<IoResult<_>>()?;
        let files: Vec<File> = ranges
            .iter()
            .map(|e| OpenOptions::new().read(true).open(e.path))
            .collect::<IoResult<_>>()?;

        // Per-request state: (bytes_done, range).
        let mut done = vec![0usize; n];
        let mut ring = Uring::new(self.options.ring_depth)?;

        // How many ops are currently in the ring.
        let mut in_flight: u32 = 0;
        // Next request to submit.
        let mut next_submit = 0usize;

        loop {
            // Submit as many as the ring can hold.
            while in_flight < self.options.ring_depth && next_submit < n {
                let idx = next_submit;
                let range = ranges[idx].range;
                let so_far = done[idx];
                let remaining = range.len_usize()? - so_far;
                if remaining == 0 {
                    next_submit += 1;
                    continue;
                }
                let len = remaining.min(MAX_IO_LEN) as u32;
                // SAFETY: `bufs[idx]` is allocated above with `range.len()` bytes.
                // `so_far < range.len()` is guaranteed by the `remaining > 0` check.
                // The ring holds a live reference to this memory until the completion
                // is drained below — `bufs` outlives the ring borrow because both
                // are owned in this stack frame.
                let ptr = unsafe { bufs[idx].as_mut_ptr().add(so_far) };
                let entry = opcode::Read::new(types::Fd(files[idx].as_raw_fd()), ptr, len)
                    .offset(range.start() + so_far as u64)
                    .build()
                    .user_data(idx as u64);
                {
                    let mut sq = ring.submission();
                    // SAFETY: see pointer comment above.
                    unsafe {
                        sq.push(&entry)
                            .map_err(|_| Error::other("io_uring SQ full"))?;
                    }
                }
                in_flight += 1;
                next_submit += 1;
            }

            if in_flight == 0 {
                break;
            }

            ring.submit_and_wait(1)?;

            // Drain all available completions.
            let cq: Vec<_> = ring.completion().collect();
            for cqe in cq {
                in_flight -= 1;
                let idx = cqe.user_data() as usize;
                let result = cqe.result();
                if result < 0 {
                    return Err(Error::from_raw_os_error(-result));
                }
                let n_read = result as usize;
                if n_read == 0 {
                    return Err(Error::new(ErrorKind::UnexpectedEof, "short io_uring read"));
                }
                done[idx] += n_read;
                // If this request is not yet complete, resubmit it.
                let range = ranges[idx].range;
                if done[idx] < range.len_usize()? {
                    // Back up next_submit so it re-submits this index.
                    if next_submit > idx {
                        next_submit = idx;
                    }
                }
            }
        }

        // All ranges complete — assemble results.
        let results = bufs
            .into_iter()
            .enumerate()
            .map(|(request_index, buf)| RangeRead {
                request_index,
                range: ranges[request_index].range,
                bytes: Arc::from(buf),
            })
            .collect();
        Ok(results)
    }

    fn write_file(&self, path: &Path, data: &[u8]) -> IoResult<()> {
        Self::ensure_available()?;
        let file = Self::open_create_truncate(path)?;
        self.write_exact_at(&file, 0, data)
    }

    fn write_positioned_file(
        &self,
        path: &Path,
        len: u64,
        writes: WriteSlices<'_>,
    ) -> IoResult<()> {
        Self::ensure_available()?;
        let file = Self::open_create_truncate(path)?;
        file.set_len(len)?;
        self.ring_batch_writes(&file, writes.as_slice())
    }

    fn write_at(&self, path: &Path, offset: u64, data: &[u8]) -> IoResult<()> {
        Self::ensure_available()?;
        let file = Self::open_write_existing(path)?;
        self.write_exact_at(&file, offset, data)
    }

    fn write_slices(&self, path: &Path, writes: WriteSlices<'_>) -> IoResult<()> {
        if writes.is_empty() {
            return Ok(());
        }
        Self::ensure_available()?;
        let file = Self::open_write_existing(path)?;
        self.ring_batch_writes(&file, writes.as_slice())
    }

    fn sync_data(&self, path: &Path) -> IoResult<()> {
        Self::ensure_available()?;
        OpenOptions::new().write(true).open(path)?.sync_data()
    }

    fn sync_all(&self, path: &Path) -> IoResult<()> {
        Self::ensure_available()?;
        OpenOptions::new().write(true).open(path)?.sync_all()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{BlockingIo, ByteRange, FileRange, Io, WriteSlices};
    use tempfile::TempDir;

    fn write_tmp(dir: &TempDir, name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, data).unwrap();
        path
    }

    fn skip_if_unavailable() -> bool {
        !IoUring::availability().is_available()
    }

    #[test]
    fn kind_is_io_uring() {
        assert_eq!(IoUring::new().kind(), IoKind::IoUring);
    }

    #[test]
    fn availability_reflects_kernel() {
        // Just assert it returns without panic — value depends on the host.
        let _ = IoUring::availability();
    }

    #[test]
    fn read_file_roundtrip() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let path = write_tmp(&dir, "file.bin", &data);

        let result = IoUring::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[test]
    fn read_file_empty() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "empty.bin", b"");

        let result = IoUring::new().read_file(&path).unwrap();
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

        let result = IoUring::new()
            .read_range(&path, ByteRange::from_offset_len(10, 20).unwrap())
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

        let result = IoUring::new()
            .read_range(&path, ByteRange::from_offset_len(0, 0).unwrap())
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_ranges_empty() {
        if skip_if_unavailable() {
            return;
        }
        let results = IoUring::new().read_ranges(&[]).unwrap();
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

        let entries = [FileRange::new(
            &path,
            ByteRange::from_offset_len(50, 30).unwrap(),
        )];
        let results = IoUring::new().read_ranges(&entries).unwrap();

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

        let entries = [
            FileRange::new(&path, ByteRange::from_offset_len(0, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(20, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(100, 5).unwrap()),
        ];
        let results = IoUring::new().read_ranges(&entries).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data(), &data[0..10]);
        assert_eq!(results[1].data(), &data[20..30]);
        assert_eq!(results[2].data(), &data[100..105]);
    }

    #[test]
    fn write_file_roundtrip() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let data = b"hello io_uring write";
        IoUring::new().write_file(&path, data).unwrap();
        IoUring::new().sync_all(&path).unwrap();
        let result = IoUring::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), data);
    }

    #[test]
    fn write_positioned_file_creates_exact_length() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pos.bin");
        let writes = [WriteSlice::new(0, b"HELLO"), WriteSlice::new(10, b"WORLD")];
        IoUring::new()
            .write_positioned_file(&path, 15, WriteSlices::new(&writes).unwrap())
            .unwrap();
        let result = IoUring::new().read_file(&path).unwrap();
        assert_eq!(result.len(), 15);
        assert_eq!(&result.as_ref()[0..5], b"HELLO");
        assert_eq!(&result.as_ref()[10..15], b"WORLD");
    }

    #[test]
    fn write_positioned_file_empty_batch_creates_file() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty_pos.bin");
        IoUring::new()
            .write_positioned_file(&path, 16, WriteSlices::new(&[]).unwrap())
            .unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 16);
    }

    #[test]
    fn write_slices_batches_into_existing_file() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "batch_write.bin", &[0u8; 20]);
        let writes = [WriteSlice::new(0, b"HELLO"), WriteSlice::new(15, b"WORLD")];
        IoUring::new()
            .write_slices(&path, WriteSlices::new(&writes).unwrap())
            .unwrap();
        let result = IoUring::new().read_file(&path).unwrap();
        assert_eq!(&result.as_ref()[0..5], b"HELLO");
        assert_eq!(&result.as_ref()[15..20], b"WORLD");
    }

    #[test]
    fn write_slices_empty_batch_is_noop() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let path = write_tmp(&dir, "noop.bin", b"unchanged");
        IoUring::new()
            .write_slices(&path, WriteSlices::new(&[]).unwrap())
            .unwrap();
        let result = IoUring::new().read_file(&path).unwrap();
        assert_eq!(result.as_ref(), b"unchanged");
    }

    #[test]
    fn write_slices_rejects_overlap() {
        let writes = [WriteSlice::new(0, b"AAAAA"), WriteSlice::new(3, b"BBBBB")];
        let err = WriteSlices::new(&writes).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn write_positioned_file_rejects_overlap() {
        let writes = [WriteSlice::new(0, b"AAAAA"), WriteSlice::new(3, b"BBBBB")];
        let err = WriteSlices::new(&writes).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn read_ranges_batch_preserves_all_request_indexes() {
        if skip_if_unavailable() {
            return;
        }
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        let path = write_tmp(&dir, "big.bin", &data);

        let entries: Vec<FileRange<'_>> = (0..8)
            .map(|i| FileRange::new(&path, ByteRange::from_offset_len(i * 32, 32).unwrap()))
            .collect();
        let results = IoUring::new().read_ranges(&entries).unwrap();

        assert_eq!(results.len(), 8);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.request_index, i, "request_index mismatch at slot {i}");
            let start = i * 32;
            assert_eq!(r.data(), &data[start..start + 32]);
        }
    }
}
