//! High-throughput io_uring backend for Linux.
//!
//! Best-practice design:
//! - Persistent ring per Reader/Writer
//! - Batched submissions with explicit in-flight queue management
//! - Multi-file scheduling on a single ring
//! - Full CQ draining per wakeup
//! - Alignment-aware direct I/O with buffered fallback

use super::byte::OwnedBytes;
use super::odirect::{can_use_direct_read, is_block_aligned, open_direct_read_sync};
use super::{IoResult, MAX_CHUNK_SIZE};
use io_uring::cqueue::CompletionQueue;
use io_uring::{opcode, types, IoUring};
use std::collections::HashMap;
use std::fs::File;
use std::io::Seek;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Ring depth tuned for NVMe/high-throughput workloads.
const RING_DEPTH: u32 = 256;

/// Minimum batch size before we bother with io_uring at all.
const IO_URING_MIN_BATCH_BYTES: usize = 4 * 1024 * 1024; // 4 MiB

/// Small-read threshold below which sync is faster than io_uring setup.
const SMALL_READ_THRESHOLD: usize = 1024 * 1024; // 1 MiB

/// A persistent io_uring reader that reuses a single ring across operations.
pub struct Reader {
    ring: IoUring,
    files: HashMap<PathBuf, File>,
}

impl Reader {
    pub fn new() -> IoResult<Self> {
        let ring = build_ring()?;
        Ok(Self {
            ring,
            files: HashMap::new(),
        })
    }

    /// Load an entire file using the persistent ring.
    pub fn load(&mut self, path: impl AsRef<Path>) -> IoResult<OwnedBytes> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let file_size = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;

        if file_size == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }

        if file_size < IO_URING_MIN_BATCH_BYTES {
            drop(file);
            return self.load_sync(path_ref, file_size);
        }

        if can_use_direct_read(file_size, file_size) {
            drop(file);
            self.load_direct(path_ref, file_size)
        } else {
            self.load_buffered(file, file_size)
        }
    }

    /// Load a single range.
    pub fn load_range(
        &mut self,
        path: impl AsRef<Path>,
        offset: u64,
        len: usize,
    ) -> IoResult<OwnedBytes> {
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }

        if len < SMALL_READ_THRESHOLD {
            return self.load_range_sync(path.as_ref(), offset, len);
        }

        let path_ref = path.as_ref();

        if is_block_aligned(offset, len) {
            match open_direct_read_sync(path_ref) {
                Ok(file) => return self.load_range_direct(file, offset, len),
                Err(err) if err.raw_os_error() == Some(libc::EINVAL) => {}
                Err(err) => return Err(err),
            }
        }

        let file = File::open(path_ref)?;
        self.load_range_buffered(file, offset, len)
    }

    /// Load multiple ranges from potentially multiple files in one batch.
    /// Returns results in the same order as the input requests.
    pub fn load_range_batch(
        &mut self,
        requests: &[(PathBuf, u64, usize)],
    ) -> IoResult<Vec<(Arc<[u8]>, usize, usize)>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Open all needed files first
        self.files.clear();
        for (path, _, _) in requests {
            if !self.files.contains_key(path) {
                let file = File::open(path)?;
                self.files.insert(path.clone(), file);
            }
        }

        // Allocate mutable buffers for each request
        let mut buffers: Vec<OwnedBytes> = Vec::with_capacity(requests.len());
        for (_, _, len) in requests {
            buffers.push(OwnedBytes::from_vec(vec![0u8; *len]));
        }

        let mut pending = requests.len();
        let mut submitted = 0;

        while pending > 0 {
            // Fill submission queue
            while submitted < requests.len() && self.ring.submission().capacity() > 0 {
                let (path, offset, len) = &requests[submitted];
                let file = self
                    .files
                    .get(path)
                    .ok_or_else(|| std::io::Error::other(format!("file not found: {:?}", path)))?;
                let fd = file.as_raw_fd();
                let ptr = buffers[submitted].as_mut_ptr();

                let sqe = opcode::Read::new(types::Fd(fd), ptr, *len as u32)
                    .offset(*offset)
                    .build()
                    .user_data(submitted as u64);

                unsafe {
                    if self.ring.submission().push(&sqe).is_err() {
                        break;
                    }
                }
                submitted += 1;
            }

            // Submit and wait
            self.ring.submit_and_wait(1)?;

            // Drain all completions
            let cq: CompletionQueue<'_, io_uring::cqueue::Entry> = self.ring.completion();
            for cqe in cq {
                let idx = cqe.user_data() as usize;
                let result = cqe.result();

                if result < 0 {
                    return Err(std::io::Error::other(format!(
                        "read error at request {}: {}",
                        idx, result
                    )));
                }

                let bytes_read = result as usize;
                let expected = requests[idx].2;
                if bytes_read < expected {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "short read at request {}: expected {} bytes, got {}",
                            idx, expected, bytes_read
                        ),
                    ));
                }
                pending -= 1;
            }
        }

        // Convert to (Arc<[u8]>, offset, len) format
        Ok(buffers
            .into_iter()
            .enumerate()
            .map(|(idx, buf)| {
                let (_, offset, len) = &requests[idx];
                (buf.into_shared(), *offset as usize, *len)
            })
            .collect())
    }

    fn load_sync(&self, path: &Path, file_size: usize) -> IoResult<OwnedBytes> {
        let mut file = File::open(path)?;
        let mut buf = vec![0u8; file_size];
        std::io::Read::read_exact(&mut file, &mut buf[..])?;
        Ok(OwnedBytes::from_vec(buf))
    }

    fn load_buffered(&mut self, file: File, file_size: usize) -> IoResult<OwnedBytes> {
        let mut buffer = OwnedBytes::from_vec(vec![0u8; file_size]);
        let base_ptr = buffer.as_mut_ptr();
        let fd = file.as_raw_fd();

        let chunk_size = MAX_CHUNK_SIZE;
        let chunks = file_size.div_ceil(chunk_size);

        if chunks == 0 {
            return Ok(buffer);
        }

        let mut pending = chunks;
        let mut submitted = 0;

        while pending > 0 {
            while submitted < chunks && self.ring.submission().capacity() > 0 {
                let offset = (submitted * chunk_size) as u64;
                let len = std::cmp::min(chunk_size, file_size - (submitted * chunk_size));
                let ptr = unsafe { base_ptr.add(submitted * chunk_size) };

                let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
                    .offset(offset)
                    .build()
                    .user_data(submitted as u64);

                unsafe {
                    if self.ring.submission().push(&sqe).is_err() {
                        break;
                    }
                }
                submitted += 1;
            }

            self.ring.submit_and_wait(1)?;

            let cq: CompletionQueue<'_, io_uring::cqueue::Entry> = self.ring.completion();
            for cqe in cq {
                let idx = cqe.user_data() as usize;
                let result = cqe.result();

                if result < 0 {
                    return Err(std::io::Error::other(format!(
                        "read error at chunk {}: {}",
                        idx, result
                    )));
                }

                let bytes_read = result as usize;
                let expected = std::cmp::min(chunk_size, file_size - (idx * chunk_size));
                if bytes_read < expected {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "short read at chunk {}: expected {} bytes, got {}",
                            idx, expected, bytes_read
                        ),
                    ));
                }
                pending -= 1;
            }
        }

        drop(file);
        Ok(buffer)
    }

    fn load_direct(&mut self, path: &Path, file_size: usize) -> IoResult<OwnedBytes> {
        let file = open_direct_read_sync(path)?;
        let mut buffer = OwnedBytes::from_vec(vec![0u8; file_size]);
        let base_ptr = buffer.as_mut_ptr();
        let fd = file.as_raw_fd();

        let chunk_size = MAX_CHUNK_SIZE;
        let chunks = file_size.div_ceil(chunk_size);

        if chunks == 0 {
            return Ok(buffer);
        }

        let mut pending = chunks;
        let mut submitted = 0;

        while pending > 0 {
            while submitted < chunks && self.ring.submission().capacity() > 0 {
                let offset = (submitted * chunk_size) as u64;
                let len = std::cmp::min(chunk_size, file_size - (submitted * chunk_size));
                let ptr = unsafe { base_ptr.add(submitted * chunk_size) };

                let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
                    .offset(offset)
                    .build()
                    .user_data(submitted as u64);

                unsafe {
                    if self.ring.submission().push(&sqe).is_err() {
                        break;
                    }
                }
                submitted += 1;
            }

            self.ring.submit_and_wait(1)?;

            let cq: CompletionQueue<'_, io_uring::cqueue::Entry> = self.ring.completion();
            for cqe in cq {
                let idx = cqe.user_data() as usize;
                let result = cqe.result();

                if result < 0 {
                    return Err(std::io::Error::other(format!(
                        "direct read error at chunk {}: {}",
                        idx, result
                    )));
                }

                let bytes_read = result as usize;
                let expected = std::cmp::min(chunk_size, file_size - (idx * chunk_size));
                if bytes_read < expected {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "short direct read at chunk {}: expected {} bytes, got {}",
                            idx, expected, bytes_read
                        ),
                    ));
                }
                pending -= 1;
            }
        }

        Ok(buffer)
    }

    fn load_range_sync(&self, path: &Path, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        let mut file = File::open(path)?;
        file.seek(std::io::SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len];
        std::io::Read::read_exact(&mut file, &mut buf[..])?;
        Ok(OwnedBytes::from_vec(buf))
    }

    fn load_range_buffered(&mut self, file: File, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        let mut buffer = OwnedBytes::from_vec(vec![0u8; len]);
        let ptr = buffer.as_mut_ptr();
        let fd = file.as_raw_fd();

        let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
            .offset(offset)
            .build()
            .user_data(0);

        unsafe {
            self.ring
                .submission()
                .push(&sqe)
                .map_err(|_| std::io::Error::other("submission queue is full"))?;
        }
        self.ring.submit_and_wait(1)?;

        let mut cq: CompletionQueue<'_, io_uring::cqueue::Entry> = self.ring.completion();
        let cqe = cq
            .next()
            .ok_or_else(|| std::io::Error::other("completion queue empty"))?;

        if cqe.result() < 0 {
            return Err(std::io::Error::other(format!(
                "read error: {}",
                cqe.result()
            )));
        }

        let bytes_read = cqe.result() as usize;
        if bytes_read < len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("short read: expected {} bytes, got {}", len, bytes_read),
            ));
        }

        drop(file);
        Ok(buffer)
    }

    fn load_range_direct(&mut self, file: File, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        let mut buffer = OwnedBytes::from_vec(vec![0u8; len]);
        let ptr = buffer.as_mut_ptr();
        let fd = file.as_raw_fd();

        let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
            .offset(offset)
            .build()
            .user_data(0);

        unsafe {
            self.ring
                .submission()
                .push(&sqe)
                .map_err(|_| std::io::Error::other("submission queue is full"))?;
        }
        self.ring.submit_and_wait(1)?;

        let mut cq: CompletionQueue<'_, io_uring::cqueue::Entry> = self.ring.completion();
        let cqe = cq
            .next()
            .ok_or_else(|| std::io::Error::other("completion queue empty"))?;

        if cqe.result() < 0 {
            return Err(std::io::Error::other(format!(
                "direct read error: {}",
                cqe.result()
            )));
        }

        let bytes_read = cqe.result() as usize;
        if bytes_read < len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "short direct read: expected {} bytes, got {}",
                    len, bytes_read
                ),
            ));
        }

        Ok(buffer)
    }
}

impl Default for Reader {
    fn default() -> Self {
        Self::new().expect("failed to create io_uring reader")
    }
}

/// Persistent io_uring writer.
pub struct Writer {
    ring: IoUring,
}

impl Writer {
    pub fn new() -> IoResult<Self> {
        let ring = build_ring()?;
        Ok(Self { ring })
    }

    pub fn create(path: &Path) -> IoResult<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::File::create(path)?;
        Self::new()
    }

    pub fn write_at(&mut self, path: &Path, offset: u64, data: &[u8]) -> IoResult<()> {
        let file = std::fs::OpenOptions::new().write(true).open(path)?;
        let fd = file.as_raw_fd();
        let ptr = data.as_ptr();

        let sqe = opcode::Write::new(types::Fd(fd), ptr, data.len() as u32)
            .offset(offset)
            .build()
            .user_data(0);

        unsafe {
            self.ring
                .submission()
                .push(&sqe)
                .map_err(|_| std::io::Error::other("submission queue is full"))?;
        }
        self.ring.submit_and_wait(1)?;

        let mut cq: CompletionQueue<'_, io_uring::cqueue::Entry> = self.ring.completion();
        let cqe = cq
            .next()
            .ok_or_else(|| std::io::Error::other("completion queue empty"))?;

        if cqe.result() < 0 {
            return Err(std::io::Error::other(format!(
                "write error: {}",
                cqe.result()
            )));
        }

        let bytes_written = cqe.result() as usize;
        if bytes_written < data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                format!(
                    "short write: expected {} bytes, got {}",
                    data.len(),
                    bytes_written
                ),
            ));
        }

        Ok(())
    }

    pub fn sync_all(&mut self) -> IoResult<()> {
        Ok(())
    }
}

impl Default for Writer {
    fn default() -> Self {
        Self::new().expect("failed to create io_uring writer")
    }
}

fn build_ring() -> IoResult<IoUring> {
    IoUring::builder()
        .setup_single_issuer()
        .setup_coop_taskrun()
        .setup_submit_all()
        .build(RING_DEPTH)
}
