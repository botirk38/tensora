//! High-throughput io_uring backend for Linux.
//!
//! Best-practice design:
//! - Persistent ring per Reader/Writer
//! - Batched submissions with explicit in-flight queue management
//! - Multi-file scheduling on a single ring
//! - Full CQ draining per wakeup
//! - Alignment-aware direct I/O with buffered fallback

use super::batch::{BatchResult, FlattenedResult, flatten_results};
use super::byte::OwnedBytes;
use super::odirect::{alloc_aligned, can_use_direct_read, is_block_aligned, open_direct_read_sync};
use super::{
    BatchRequest, IoResult, MAX_CHUNK_SIZE, batch::group_requests_by_file, get_buffer_pool,
};
use io_uring::cqueue::CompletionQueue;
use io_uring::{IoUring, opcode, types};
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

/// io_uring-specific chunk size. Larger than sync to reduce SQE/CQE churn.
const IO_URING_CHUNK_SIZE: usize = 512 * 1024 * 1024; // 512 MiB

const WAIT_FOR_COMPLETIONS: usize = RING_DEPTH as usize;
const COMPLETION_RING_DEPTH: u32 = RING_DEPTH * 2;
const SQPOLL_IDLE_MS: u32 = 2_000;

/// A persistent io_uring reader that reuses a single ring across operations.
pub struct Reader {
    ring: IoUring,
    files: HashMap<PathBuf, File>,
}

struct ChunkReadOp {
    plan_idx: usize,
    offset: u64,
    len: usize,
    dst_offset: usize,
}

struct PreparedLoad {
    request_idx: usize,
    file: File,
    buffer: OwnedBytes,
    base_ptr: usize,
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

    pub fn load_batch(&mut self, paths: &[PathBuf]) -> IoResult<Vec<OwnedBytes>> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }

        let mut results: Vec<Option<OwnedBytes>> = (0..paths.len()).map(|_| None).collect();
        let mut plans = Vec::with_capacity(paths.len());
        let mut ops = Vec::new();

        for (request_idx, path) in paths.iter().enumerate() {
            let file = File::open(path)?;
            let file_size = usize::try_from(file.metadata()?.len())
                .map_err(|_| std::io::Error::other("file too large"))?;

            if file_size == 0 {
                results[request_idx] = Some(OwnedBytes::Shared(Arc::new([])));
                continue;
            }

            if file_size < IO_URING_MIN_BATCH_BYTES {
                drop(file);
                results[request_idx] = Some(self.load_sync(path, file_size)?);
                continue;
            }

            let (file, mut buffer) = if can_use_direct_read(file_size, file_size) {
                drop(file);
                let mut aligned = alloc_aligned(file_size)?;
                aligned.set_len(file_size);
                (
                    open_direct_read_sync(path)?,
                    OwnedBytes::from_aligned(aligned),
                )
            } else {
                let pooled = get_buffer_pool().get(file_size);
                (file, OwnedBytes::from_pooled(pooled))
            };

            let plan_idx = plans.len();
            let base_ptr = buffer.as_mut_ptr() as usize;
            let chunk_size = IO_URING_CHUNK_SIZE.min(MAX_CHUNK_SIZE.saturating_mul(4));

            for chunk_start in (0..file_size).step_by(chunk_size) {
                let len = chunk_size.min(file_size - chunk_start);
                ops.push(ChunkReadOp {
                    plan_idx,
                    offset: chunk_start as u64,
                    len,
                    dst_offset: chunk_start,
                });
            }

            plans.push(PreparedLoad {
                request_idx,
                file,
                buffer,
                base_ptr,
            });
        }

        if !ops.is_empty() {
            let mut completed = 0usize;
            let mut submitted = 0usize;

            while completed < ops.len() {
                while submitted < ops.len() && self.ring.submission().capacity() > 0 {
                    let op = &ops[submitted];
                    let plan = &plans[op.plan_idx];
                    let fd = plan.file.as_raw_fd();
                    let ptr = (plan.base_ptr + op.dst_offset) as *mut u8;

                    let sqe = opcode::Read::new(types::Fd(fd), ptr, op.len as u32)
                        .offset(op.offset)
                        .build()
                        .user_data(submitted as u64);

                    unsafe {
                        if self.ring.submission().push(&sqe).is_err() {
                            break;
                        }
                    }
                    submitted += 1;
                }

                let wait_for = (ops.len() - completed).clamp(1, WAIT_FOR_COMPLETIONS);
                self.ring.submit_and_wait(wait_for)?;

                let cq: CompletionQueue<'_, io_uring::cqueue::Entry> = self.ring.completion();
                for cqe in cq {
                    let op_idx = cqe.user_data() as usize;
                    let result = cqe.result();

                    if result < 0 {
                        return Err(std::io::Error::other(format!(
                            "read error at batch op {}: {}",
                            op_idx, result
                        )));
                    }

                    let bytes_read = result as usize;
                    let expected = ops[op_idx].len;
                    if bytes_read < expected {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            format!(
                                "short read at batch op {}: expected {} bytes, got {}",
                                op_idx, expected, bytes_read
                            ),
                        ));
                    }

                    completed += 1;
                }
            }
        }

        for plan in plans {
            results[plan.request_idx] = Some(plan.buffer);
        }

        results
            .into_iter()
            .map(|result| result.ok_or_else(|| std::io::Error::other("missing batch result")))
            .collect()
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
        requests: &[BatchRequest],
    ) -> IoResult<Vec<FlattenedResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Group requests by file
        let grouped = group_requests_by_file(requests);

        // Open all needed files first
        self.files.clear();
        for path in grouped.keys() {
            if !self.files.contains_key(path) {
                let file = File::open(path)?;
                self.files.insert(path.clone(), file);
            }
        }

        let mut planned = Vec::with_capacity(requests.len());
        for (path, mut reqs) in grouped {
            reqs.sort_unstable_by_key(|req| req.offset);
            for req in reqs {
                planned.push((path.clone(), req));
            }
        }

        // Allocate buffers for each request
        let mut buffers: Vec<OwnedBytes> = Vec::with_capacity(requests.len());
        for (_, _, len) in requests {
            buffers.push(OwnedBytes::from_vec(vec![0u8; *len]));
        }

        let mut pending = planned.len();
        let mut submitted = 0;
        let mut grouped_results: Vec<Vec<BatchResult>> = vec![Vec::new()];

        while pending > 0 {
            // Fill submission queue
            while submitted < planned.len() && self.ring.submission().capacity() > 0 {
                let (path, req) = &planned[submitted];
                let file = self
                    .files
                    .get(path)
                    .ok_or_else(|| std::io::Error::other(format!("file not found: {:?}", path)))?;
                let fd = file.as_raw_fd();
                let ptr = buffers[req.idx].as_mut_ptr();

                let sqe = opcode::Read::new(types::Fd(fd), ptr, req.len as u32)
                    .offset(req.offset)
                    .build()
                    .user_data(req.idx as u64);

                unsafe {
                    if self.ring.submission().push(&sqe).is_err() {
                        break;
                    }
                }
                submitted += 1;
            }

            if submitted == 0 {
                break;
            }

            let wait_for = pending.clamp(1, WAIT_FOR_COMPLETIONS);
            self.ring.submit_and_wait(wait_for)?;

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
                grouped_results[0].push((idx, buffers[idx].as_ref().into(), 0, expected));
                pending -= 1;
            }
        }

        Ok(flatten_results(grouped_results))
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

        let chunk_size = IO_URING_CHUNK_SIZE.min(MAX_CHUNK_SIZE.saturating_mul(4));
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

            let wait_for = pending.clamp(1, WAIT_FOR_COMPLETIONS);
            self.ring.submit_and_wait(wait_for)?;

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
        let mut aligned = alloc_aligned(file_size)?;
        aligned.set_len(file_size);
        let mut buffer = OwnedBytes::from_aligned(aligned);
        let base_ptr = buffer.as_mut_ptr();
        let fd = file.as_raw_fd();

        let chunk_size = IO_URING_CHUNK_SIZE.min(MAX_CHUNK_SIZE.saturating_mul(4));
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

            let wait_for = pending.clamp(1, WAIT_FOR_COMPLETIONS);
            self.ring.submit_and_wait(wait_for)?;

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
        let mut aligned = alloc_aligned(len)?;
        aligned.set_len(len);
        let mut buffer = OwnedBytes::from_aligned(aligned);
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
        let mut aligned = alloc_aligned(len)?;
        aligned.set_len(len);
        let mut buffer = OwnedBytes::from_aligned(aligned);
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
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
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
    match build_ring_with_sqpoll() {
        Ok(ring) => return Ok(ring),
        Err(err)
            if matches!(
                err.raw_os_error(),
                Some(libc::EINVAL | libc::EPERM | libc::ENOSYS)
            ) => {}
        Err(err) => return Err(err),
    }

    IoUring::builder()
        .setup_single_issuer()
        .setup_coop_taskrun()
        .setup_submit_all()
        .setup_cqsize(COMPLETION_RING_DEPTH)
        .build(RING_DEPTH)
}

fn build_ring_with_sqpoll() -> IoResult<IoUring> {
    IoUring::builder()
        .setup_single_issuer()
        .setup_coop_taskrun()
        .setup_submit_all()
        .setup_cqsize(COMPLETION_RING_DEPTH)
        .setup_sqpoll(SQPOLL_IDLE_MS)
        .build(RING_DEPTH)
}
