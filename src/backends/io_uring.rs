//! High-throughput io_uring backend for Linux.
//!
//! Best-practice design:
//! - Persistent ring per Reader/Writer
//! - Batched submissions with explicit in-flight queue management
//! - Multi-file scheduling on a single ring
//! - Full CQ draining per wakeup
//! - Alignment-aware direct I/O with buffered fallback

use super::batch::{
    BatchResult, CoalescedRequestGroup, FlattenedResult, coalesce_requests, flatten_results,
};
use super::byte::OwnedBytes;
use super::odirect::{alloc_aligned, can_use_direct_read, is_block_aligned, open_direct_read};
use super::{
    BackendKind, BatchRequest, IoResult, batch::group_requests_by_file, chunk_budget,
    file_chunk_plan, get_buffer_pool, range_batch_plan,
};
use io_uring::cqueue::CompletionQueue;
use io_uring::{IoUring, opcode, types};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

/// Maximum ring depth for NVMe/high-throughput workloads.
const MAX_RING_DEPTH: u32 = 256;
const MIN_RING_DEPTH: u32 = 32;

const DEFAULT_SQPOLL_IDLE_MS: u32 = 2_000;

/// A persistent io_uring reader that reuses a single ring across operations.
pub struct Reader {
    ring: Option<IoUring>,
    files: HashMap<PathBuf, File>,
    ring_depth_override: Option<u32>,
    cq_depth_override: Option<u32>,
    enable_sqpoll: bool,
    sqpoll_idle_ms: u32,
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

#[derive(Clone, Copy)]
struct ReaderConfig {
    ring_depth_override: Option<u32>,
    cq_depth_override: Option<u32>,
    enable_sqpoll: bool,
    sqpoll_idle_ms: u32,
}

impl Reader {
    pub fn new() -> Self {
        Self {
            ring: None,
            files: HashMap::new(),
            ring_depth_override: None,
            cq_depth_override: None,
            enable_sqpoll: true,
            sqpoll_idle_ms: DEFAULT_SQPOLL_IDLE_MS,
        }
    }

    pub fn with_ring_depth(mut self, depth: u32) -> Self {
        self.ring_depth_override = Some(depth.clamp(MIN_RING_DEPTH, MAX_RING_DEPTH));
        self
    }

    pub fn with_cq_depth(mut self, depth: u32) -> Self {
        self.cq_depth_override = Some(depth.clamp(MIN_RING_DEPTH, MAX_RING_DEPTH * 2));
        self
    }

    pub fn with_sqpoll(mut self, enabled: bool) -> Self {
        self.enable_sqpoll = enabled;
        self
    }

    pub fn with_sqpoll_idle_ms(mut self, idle_ms: u32) -> Self {
        self.sqpoll_idle_ms = idle_ms.max(1);
        self
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

        self.ensure_ring_for_file_load(file_size, 1, file_size)?;

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

        let total_bytes: usize = paths
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .filter_map(|meta| usize::try_from(meta.len()).ok())
            .sum();
        let max_file_bytes = paths
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .filter_map(|meta| usize::try_from(meta.len()).ok())
            .max()
            .unwrap_or(0);

        let worker_count = file_worker_count(paths.len(), total_bytes, max_file_bytes);
        
        if worker_count > 1 {
            let indexed_paths: Vec<_> = paths.iter().cloned().enumerate().collect();
            let mut indexed_results = Vec::with_capacity(paths.len());
            let config = self.config();

            thread::scope(|scope| -> IoResult<()> {
                let mut handles = Vec::new();
                for chunk in indexed_paths.chunks(indexed_paths.len().div_ceil(worker_count)) {
                    let chunk = chunk.to_vec();
                    handles.push(scope.spawn(move || {
                        let mut reader = Reader::from_config(config);
                        reader.load_batch_indexed(&chunk)
                    }));
                }

                for handle in handles {
                    indexed_results.extend(
                        handle
                            .join()
                            .map_err(|_| std::io::Error::other("io_uring worker panicked"))??,
                    );
                }
                Ok(())
            })?;

            indexed_results.sort_unstable_by_key(|(idx, _)| *idx);
            return indexed_results
                .into_iter()
                .map(|(_, bytes)| Ok(bytes))
                .collect();
        }

        self.ensure_ring_for_file_load(total_bytes, paths.len(), max_file_bytes)?;

        self.load_batch_indexed(&paths.iter().cloned().enumerate().collect::<Vec<_>>())
            .map(|results| {
                results
                    .into_iter()
                    .map(|(_, bytes)| bytes)
                    .collect::<Vec<_>>()
            })
    }

    fn load_batch_indexed(
        &mut self,
        paths: &[(usize, PathBuf)],
    ) -> IoResult<Vec<(usize, OwnedBytes)>> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }

        let total_bytes: usize = paths
            .iter()
            .filter_map(|(_, path)| std::fs::metadata(path).ok())
            .filter_map(|meta| usize::try_from(meta.len()).ok())
            .sum();
        let load_plan = file_chunk_plan(total_bytes.max(1), BackendKind::IoUring);
        let max_file_bytes = paths
            .iter()
            .filter_map(|(_, path)| std::fs::metadata(path).ok())
            .filter_map(|meta| usize::try_from(meta.len()).ok())
            .max()
            .unwrap_or(0);

        self.ensure_ring_for_file_load(total_bytes, paths.len(), max_file_bytes)?;

        let mut results: Vec<Option<(usize, OwnedBytes)>> =
            (0..paths.len()).map(|_| None).collect();
        let mut plans = Vec::with_capacity(paths.len());
        let mut ops = Vec::new();

        for (slot_idx, (request_idx, path)) in paths.iter().enumerate() {
            let file = File::open(path)?;
            let file_size = usize::try_from(file.metadata()?.len())
                .map_err(|_| std::io::Error::other("file too large"))?;

            if file_size == 0 {
                results[slot_idx] = Some((*request_idx, OwnedBytes::Shared(Arc::new([]))));
                continue;
            }

            let (file, mut buffer) = if can_use_direct_read(file_size, file_size) {
                drop(file);
                let mut aligned = alloc_aligned(file_size)?;
                aligned.set_len(file_size);
                (
                    open_direct_read(path)?,
                    OwnedBytes::from_aligned(aligned),
                )
            } else {
                let pooled = get_buffer_pool().get(file_size);
                (file, OwnedBytes::from_pooled(pooled))
            };

            let plan_idx = plans.len();
            let base_ptr = buffer.as_mut_ptr() as usize;
            let chunk_size = file_chunk_plan(file_size, BackendKind::IoUring).chunk_size;

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
                request_idx: slot_idx,
                file,
                buffer,
                base_ptr,
            });
        }

        if !ops.is_empty() {
            let mut completed = 0usize;
            let mut submitted = 0usize;
            let ring = self.ring_mut()?;

            let mut max_inflight = 0usize;

            while completed < ops.len() {
                while submitted < ops.len() && ring.submission().capacity() > 0 {
                    let op = &ops[submitted];
                    let plan = &plans[op.plan_idx];
                    let fd = plan.file.as_raw_fd();
                    let ptr = (plan.base_ptr + op.dst_offset) as *mut u8;

                    let sqe = opcode::Read::new(types::Fd(fd), ptr, op.len as u32)
                        .offset(op.offset)
                        .build()
                        .user_data(submitted as u64);

                    unsafe {
                        if ring.submission().push(&sqe).is_err() {
                            break;
                        }
                    }
                    submitted += 1;
                    let inflight = submitted - completed;
                    if inflight > max_inflight {
                        max_inflight = inflight;
                    }
                }

                let wait_for = load_plan.wait_for.min(ops.len() - completed).max(1);
                ring.submit_and_wait(wait_for)?;

                let cq: CompletionQueue<'_, io_uring::cqueue::Entry> = ring.completion();
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
            let request_idx = paths[plan.request_idx].0;
            results[plan.request_idx] = Some((request_idx, plan.buffer));
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

        self.ensure_ring_for_range_load(len, 1, len)?;

        let path_ref = path.as_ref();

        if is_block_aligned(offset, len) && let Ok(file) = open_direct_read(path_ref) {
            return self.load_range_buffered(file, offset, len);
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

        let total_bytes: usize = requests.iter().map(|(_, _, len)| *len).sum();
        let batch_plan = range_batch_plan(requests.len(), total_bytes, BackendKind::IoUring);
        let avg_bytes = total_bytes.div_ceil(requests.len()).max(1);
        let coalesced = coalesce_requests(grouped, batch_plan.coalesce_window_bytes);

        let worker_count = range_worker_count(&coalesced, total_bytes);
        if worker_count > 1 {
            let config = self.config();
            let chunk_size = coalesced.len().div_ceil(worker_count);
            let mut grouped_results = Vec::new();

            thread::scope(|scope| -> IoResult<()> {
                let mut handles = Vec::new();
                for chunk in coalesced.chunks(chunk_size) {
                    let chunk = chunk.to_vec();
                    handles.push(scope.spawn(move || {
                        let mut reader = Reader::from_config(config);
                        reader.load_range_groups(&chunk)
                    }));
                }

                for handle in handles {
                    grouped_results.push(
                        handle
                            .join()
                            .map_err(|_| std::io::Error::other("io_uring worker panicked"))??,
                    );
                }
                Ok(())
            })?;

            return Ok(flatten_results(grouped_results));
        }

        self.ensure_ring_for_range_load(total_bytes, requests.len(), avg_bytes)?;

        Ok(flatten_results(vec![self.load_range_groups(&coalesced)?]))
    }

    fn load_range_groups(
        &mut self,
        groups: &[CoalescedRequestGroup],
    ) -> IoResult<Vec<BatchResult>> {
        if groups.is_empty() {
            return Ok(Vec::new());
        }

        self.files.clear();
        for group in groups {
            if !self.files.contains_key(&group.path) {
                let file = File::open(&group.path)?;
                self.files.insert(group.path.clone(), file);
            }
        }

        let total_bytes: usize = groups.iter().map(|group| group.len).sum();
        let total_members: usize = groups.iter().map(|group| group.members.len()).sum();
        let avg_bytes = total_bytes.div_ceil(groups.len()).max(1);
        let batch_plan = range_batch_plan(
            total_members.max(1),
            total_bytes.max(1),
            BackendKind::IoUring,
        );

        self.ensure_ring_for_range_load(total_bytes, total_members.max(1), avg_bytes)?;

        let mut planned: Vec<(RawFd, usize, u64, usize)> = Vec::with_capacity(groups.len());
        let mut buffers: Vec<OwnedBytes> = Vec::with_capacity(groups.len());
        let mut members = Vec::with_capacity(groups.len());
        for group in groups {
            let fd = self
                .files
                .get(&group.path)
                .ok_or_else(|| std::io::Error::other(format!("file not found: {:?}", group.path)))?
                .as_raw_fd();
            planned.push((fd, members.len(), group.offset, group.len));
            buffers.push(OwnedBytes::from_vec(vec![0u8; group.len]));
            members.push(group.members.clone());
        }

        let mut pending = planned.len();
        let mut submitted = 0;
        let mut completed = 0usize;
        let ring = self.ring_mut()?;
        let mut results = Vec::with_capacity(total_members);

        while pending > 0 {
            // Fill submission queue
            while submitted < planned.len()
                && ring.submission().capacity() > 0
                && submitted.saturating_sub(completed) < batch_plan.target_inflight
            {
                let (fd, buffer_idx, offset, len) = planned[submitted];
                let ptr = buffers[buffer_idx].as_mut_ptr();

                let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
                    .offset(offset)
                    .build()
                    .user_data(buffer_idx as u64);

                unsafe {
                    if ring.submission().push(&sqe).is_err() {
                        break;
                    }
                }
                submitted += 1;
            }

            if submitted == 0 {
                break;
            }

            let wait_for = pending.min(batch_plan.wait_for).max(1);
            ring.submit_and_wait(wait_for)?;

            // Drain all completions
            let cq: CompletionQueue<'_, io_uring::cqueue::Entry> = ring.completion();
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
                let expected = planned[idx].3;
                if bytes_read < expected {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "short read at request {}: expected {} bytes, got {}",
                            idx, expected, bytes_read
                        ),
                    ));
                }
                let backing: Arc<[u8]> = buffers[idx].as_ref().into();
                for member in &members[idx] {
                    let start = member.relative_offset;
                    let end = start.saturating_add(member.len);
                    let slice = backing.get(start..end).ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            format!(
                                "coalesced slice out of bounds for request {}: {}..{} of {}",
                                member.idx,
                                start,
                                end,
                                backing.len()
                            ),
                        )
                    })?;
                    results.push((member.idx, Arc::<[u8]>::from(slice), 0, member.len));
                }
                pending -= 1;
                completed += 1;
            }
        }

        Ok(results)
    }

    fn load_buffered(&mut self, file: File, file_size: usize) -> IoResult<OwnedBytes> {
        let mut buffer = OwnedBytes::from_vec(vec![0u8; file_size]);
        let base_ptr = buffer.as_mut_ptr();
        let fd = file.as_raw_fd();

        self.read_chunks(fd, base_ptr, file_size)?;

        drop(file);
        Ok(buffer)
    }

    fn load_direct(&mut self, path: &Path, file_size: usize) -> IoResult<OwnedBytes> {
        let file = open_direct_read(path)?;
        let mut aligned = alloc_aligned(file_size)?;
        aligned.set_len(file_size);
        let mut buffer = OwnedBytes::from_aligned(aligned);
        let base_ptr = buffer.as_mut_ptr();
        let fd = file.as_raw_fd();

        self.read_chunks(fd, base_ptr, file_size)?;

        Ok(buffer)
    }

    fn read_chunks(&mut self, fd: RawFd, base_ptr: *mut u8, file_size: usize) -> IoResult<()> {
        let plan = file_chunk_plan(file_size, BackendKind::IoUring);
        let chunk_size = plan.chunk_size;
        let chunks = plan.chunk_count;

        if chunks == 0 {
            return Ok(());
        }

        let mut pending = chunks;
        let mut submitted = 0;
        let ring = self.ring_mut()?;

        while pending > 0 {
            while submitted < chunks && ring.submission().capacity() > 0 {
                let offset = (submitted * chunk_size) as u64;
                let len = chunk_size.min(file_size - submitted * chunk_size);
                let ptr = unsafe { base_ptr.add(submitted * chunk_size) };

                let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
                    .offset(offset)
                    .build()
                    .user_data(submitted as u64);

                unsafe {
                    if ring.submission().push(&sqe).is_err() {
                        break;
                    }
                }
                submitted += 1;
            }

            let wait_for = pending.min(plan.wait_for).max(1);
            ring.submit_and_wait(wait_for)?;

            let cq: CompletionQueue<'_, io_uring::cqueue::Entry> = ring.completion();
            for cqe in cq {
                let idx = cqe.user_data() as usize;
                let result = cqe.result();

                if result < 0 {
                    return Err(std::io::Error::other(format!("read error at chunk {}", idx)));
                }

                let bytes_read = result as usize;
                let expected = chunk_size.min(file_size - idx * chunk_size);
                if bytes_read < expected {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!("short read at chunk {}: expected {} bytes, got {}", idx, expected, bytes_read),
                    ));
                }
                pending -= 1;
            }
        }

        Ok(())
    }

    fn load_range_buffered(&mut self, file: File, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        let mut aligned = alloc_aligned(len)?;
        aligned.set_len(len);
        let mut buffer = OwnedBytes::from_aligned(aligned);
        let ptr = buffer.as_mut_ptr();
        let fd = file.as_raw_fd();
        let ring = self.ring_mut()?;

        let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
            .offset(offset)
            .build()
            .user_data(0);

        unsafe {
            ring.submission()
                .push(&sqe)
                .map_err(|_| std::io::Error::other("submission queue is full"))?;
        }
        ring.submit_and_wait(1)?;

        let mut cq: CompletionQueue<'_, io_uring::cqueue::Entry> = ring.completion();
        let cqe = cq
            .next()
            .ok_or_else(|| std::io::Error::other("completion queue empty"))?;

        if cqe.result() < 0 {
            return Err(std::io::Error::other(format!("read error: {}", cqe.result())));
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

    fn ring_mut(&mut self) -> IoResult<&mut IoUring> {
        self.ring
            .as_mut()
            .ok_or_else(|| std::io::Error::other("io_uring ring not initialized"))
    }

    fn ensure_ring_for_file_load(
        &mut self,
        total_bytes: usize,
        file_count: usize,
        max_file_bytes: usize,
    ) -> IoResult<()> {
        if self.ring.is_some() {
            return Ok(());
        }

        let default_ring_depth = if file_count <= 1 {
            if max_file_bytes <= 512 * 1024 * 1024 {
                32
            } else if max_file_bytes <= 4 * 1024 * 1024 * 1024 {
                64
            } else {
                128
            }
        } else if total_bytes <= 4 * 1024 * 1024 * 1024 {
            64
        } else if total_bytes <= 32 * 1024 * 1024 * 1024 {
            128
        } else {
            MAX_RING_DEPTH
        };

        self.ensure_ring(default_ring_depth)
    }

    fn ensure_ring_for_range_load(
        &mut self,
        total_bytes: usize,
        request_count: usize,
        avg_request_bytes: usize,
    ) -> IoResult<()> {
        if self.ring.is_some() {
            return Ok(());
        }

        let default_ring_depth = if request_count <= 1 {
            32
        } else if avg_request_bytes <= 256 * 1024 {
            128
        } else if total_bytes <= 64 * 1024 * 1024 {
            64
        } else {
            96
        };

        self.ensure_ring(default_ring_depth)
    }

    fn ensure_ring(&mut self, default_ring_depth: u32) -> IoResult<()> {
        if self.ring.is_some() {
            return Ok(());
        }

        let ring_depth = self
            .ring_depth_override
            .unwrap_or(default_ring_depth)
            .clamp(MIN_RING_DEPTH, MAX_RING_DEPTH);
        let cq_depth = self
            .cq_depth_override
            .unwrap_or(ring_depth.saturating_mul(2))
            .clamp(ring_depth, MAX_RING_DEPTH * 2);

        self.ring = Some(build_ring_with_config(
            ring_depth,
            cq_depth,
            self.enable_sqpoll,
            self.sqpoll_idle_ms,
        )?);
        Ok(())
    }

    fn config(&self) -> ReaderConfig {
        ReaderConfig {
            ring_depth_override: self.ring_depth_override,
            cq_depth_override: self.cq_depth_override,
            enable_sqpoll: self.enable_sqpoll,
            sqpoll_idle_ms: self.sqpoll_idle_ms,
        }
    }

    fn from_config(config: ReaderConfig) -> Self {
        Self {
            ring: None,
            files: HashMap::new(),
            ring_depth_override: config.ring_depth_override,
            cq_depth_override: config.cq_depth_override,
            enable_sqpoll: config.enable_sqpoll,
            sqpoll_idle_ms: config.sqpoll_idle_ms,
        }
    }
}

impl Default for Reader {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistent io_uring writer.
pub struct Writer {
    ring: IoUring,
}

impl Writer {
    pub fn new() -> IoResult<Self> {
        let ring = build_default_ring()?;
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

fn build_default_ring() -> IoResult<IoUring> {
    build_ring_with_config(
        MAX_RING_DEPTH,
        MAX_RING_DEPTH * 2,
        true,
        DEFAULT_SQPOLL_IDLE_MS,
    )
}

fn build_ring_with_config(
    ring_depth: u32,
    cq_depth: u32,
    enable_sqpoll: bool,
    sqpoll_idle_ms: u32,
) -> IoResult<IoUring> {
    if enable_sqpoll {
        match build_ring_with_sqpoll(ring_depth, cq_depth, sqpoll_idle_ms) {
            Ok(ring) => return Ok(ring),
            Err(err)
                if matches!(
                    err.raw_os_error(),
                    Some(libc::EINVAL | libc::EPERM | libc::ENOSYS)
                ) => {}
            Err(err) => return Err(err),
        }
    }

    IoUring::builder()
        .setup_single_issuer()
        .setup_coop_taskrun()
        .setup_submit_all()
        .setup_cqsize(cq_depth)
        .build(ring_depth)
}

fn build_ring_with_sqpoll(
    ring_depth: u32,
    cq_depth: u32,
    sqpoll_idle_ms: u32,
) -> IoResult<IoUring> {
    IoUring::builder()
        .setup_single_issuer()
        .setup_coop_taskrun()
        .setup_submit_all()
        .setup_cqsize(cq_depth)
        .setup_sqpoll(sqpoll_idle_ms)
        .build(ring_depth)
}

fn file_worker_count(file_count: usize, total_bytes: usize, max_file_bytes: usize) -> usize {
    if file_count <= 1 || total_bytes < 2 * 1024 * 1024 * 1024 {
        return 1;
    }

    let base_workers = chunk_budget().clamp(1, 8).min(file_count);
    
    let workers = if max_file_bytes >= 512 * 1024 * 1024 {
        (base_workers * 2).min(16)
    } else {
        base_workers.min(8)
    };
    
    workers.max(1)
}

fn range_worker_count(groups: &[CoalescedRequestGroup], total_bytes: usize) -> usize {
    let file_count = groups
        .iter()
        .map(|group| &group.path)
        .collect::<std::collections::HashSet<_>>()
        .len();

    if file_count <= 1 || groups.len() <= 1 || total_bytes < 512 * 1024 * 1024 {
        return 1;
    }

    chunk_budget().clamp(1, 8).min(file_count).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::SyncReader;
    use std::fs;
    use tempfile::TempDir;

    fn write_file(path: &Path, len: usize, seed: u8) {
        let data: Vec<u8> = (0..len)
            .map(|i| seed.wrapping_add((i % 251) as u8))
            .collect();
        fs::write(path, data).unwrap();
    }

    #[test]
    fn load_matches_sync_reader_for_large_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("large.bin");
        write_file(&path, 5 * 1024 * 1024, 7);

        let mut sync = SyncReader::new();
        let expected = sync.load(&path).unwrap();

        let mut uring = Reader::new().with_sqpoll(false);
        let actual = uring.load(&path).unwrap();

        assert_eq!(actual.as_ref(), expected.as_ref());
    }

    #[test]
    fn load_batch_preserves_order_and_bytes() {
        let dir = TempDir::new().unwrap();
        let path_a = dir.path().join("a.bin");
        let path_b = dir.path().join("b.bin");
        write_file(&path_a, 5 * 1024 * 1024, 3);
        write_file(&path_b, 6 * 1024 * 1024, 9);

        let mut sync = SyncReader::new();
        let expected_a = sync.load(&path_a).unwrap();
        let expected_b = sync.load(&path_b).unwrap();

        let mut uring = Reader::new().with_sqpoll(false);
        let results = uring.load_batch(&[path_a.clone(), path_b.clone()]).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].as_ref(), expected_a.as_ref());
        assert_eq!(results[1].as_ref(), expected_b.as_ref());
    }

    #[test]
    fn load_range_batch_matches_sync_reader() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ranges.bin");
        write_file(&path, 6 * 1024 * 1024, 13);

        let requests = vec![
            (path.clone(), 0, 512 * 1024),
            (path.clone(), 520 * 1024, 256 * 1024),
            (path.clone(), 2 * 1024 * 1024, 768 * 1024),
        ];

        let mut sync = SyncReader::new();
        let expected = sync.load_range_batch(&requests).unwrap();

        let mut uring = Reader::new().with_sqpoll(false);
        let actual = uring.load_range_batch(&requests).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual_item, expected_item) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_item.1, expected_item.1);
            assert_eq!(actual_item.2, expected_item.2);
            assert_eq!(actual_item.0.as_ref(), expected_item.0.as_ref());
        }
    }

    #[test]
    fn manual_ring_overrides_are_clamped() {
        let reader = Reader::new()
            .with_ring_depth(1)
            .with_cq_depth(u32::MAX)
            .with_sqpoll_idle_ms(0);

        assert_eq!(reader.ring_depth_override, Some(MIN_RING_DEPTH));
        assert_eq!(reader.cq_depth_override, Some(MAX_RING_DEPTH * 2));
        assert_eq!(reader.sqpoll_idle_ms, 1);
    }
}
