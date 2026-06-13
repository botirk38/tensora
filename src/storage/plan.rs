//! I/O planning for storage engines.
//!
//! Computes chunk sizes, inflight counts, and coalesce windows for file and
//! range-batch reads. All functions are pure and take no global state —
//! the caller passes the engine kind so the plan is tuned appropriately.

// ============================================================================
// Constants
// ============================================================================

pub const MAX_SINGLE_READ: usize = 512 * 1024 * 1024;
pub const MAX_CHUNK_SIZE: usize = 128 * 1024 * 1024;
pub const MIN_CHUNK_SIZE: usize = 32 * 1024 * 1024;
pub const MAX_IO_URING_CHUNK_SIZE: usize = 512 * 1024 * 1024;
pub const MAX_IO_URING_DEPTH: usize = 256;

// ============================================================================
// EngineKind — local enum used only for planning decisions
// ============================================================================

/// Identifies the engine so the planner can tune parameters appropriately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Sync,
    Tokio,
    IoUring,
}

// ============================================================================
// ChunkPlan
// ============================================================================

/// A single chunk of a file read: byte offset and byte length.
#[derive(Debug, Clone, Copy)]
pub struct ChunkPlan {
    /// Byte offset within the file.
    pub offset: u64,
    /// Number of bytes in this chunk.
    pub len: usize,
}

// ============================================================================
// FileIoPlan
// ============================================================================

/// Parameters for splitting a file read into parallel chunks.
#[derive(Debug, Clone, Copy)]
pub struct FileIoPlan {
    /// Target size of each chunk in bytes.
    pub chunk_size: usize,
    /// Total number of chunks needed to cover the file.
    pub chunk_count: usize,
    /// How many chunks to submit concurrently.
    pub target_inflight: usize,
    /// How many completions to wait for before submitting more.
    pub wait_for: usize,
}

// ============================================================================
// RangeBatchPlan
// ============================================================================

/// Parameters for dispatching a batch of range reads in parallel.
#[derive(Debug, Clone, Copy)]
pub struct RangeBatchPlan {
    /// How many ranges to read concurrently.
    pub target_inflight: usize,
    /// How many completions to wait for before submitting more.
    pub wait_for: usize,
    /// Adjacent ranges within this many bytes are coalesced into one read.
    pub coalesce_window_bytes: usize,
}

// ============================================================================
// Planning functions
// ============================================================================

/// Returns the number of available CPU threads, clamped to at least 1.
#[inline]
#[must_use]
pub fn chunk_budget() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1)
}

#[inline]
fn clamp_chunk_size(size: usize, max_size: usize) -> usize {
    size.clamp(MIN_CHUNK_SIZE, max_size)
}

/// Compute chunk parameters for reading a whole file with the given engine.
///
/// Returns a sensible `FileIoPlan` even when `file_size` is zero.
#[must_use]
pub fn file_chunk_plan(file_size: usize, engine: EngineKind) -> FileIoPlan {
    if file_size == 0 {
        return FileIoPlan {
            chunk_size: MIN_CHUNK_SIZE,
            chunk_count: 1,
            target_inflight: 1,
            wait_for: 1,
        };
    }

    let cpus = chunk_budget();
    match engine {
        EngineKind::Sync => {
            let target_parallelism = cpus.clamp(1, 8);
            let raw_chunk = file_size.div_ceil(target_parallelism.max(1));
            let chunk_size = clamp_chunk_size(raw_chunk, MAX_CHUNK_SIZE);
            let chunk_count = file_size.div_ceil(chunk_size).max(1);
            let target_inflight = chunk_count.min(target_parallelism).max(1);
            FileIoPlan { chunk_size, chunk_count, target_inflight, wait_for: target_inflight }
        }
        EngineKind::Tokio => {
            let target_parallelism = cpus.clamp(1, 8);
            let raw_chunk = file_size.div_ceil((target_parallelism * 2).max(1));
            let chunk_size = clamp_chunk_size(raw_chunk, MAX_CHUNK_SIZE);
            let chunk_count = file_size.div_ceil(chunk_size).max(1);
            let target_inflight = chunk_count.min(target_parallelism * 2).max(1);
            let wait_for = target_inflight.min(target_parallelism).max(1);
            FileIoPlan { chunk_size, chunk_count, target_inflight, wait_for }
        }
        EngineKind::IoUring => {
            let target_depth = if file_size < 64 * 1024 * 1024 {
                16
            } else if file_size < 512 * 1024 * 1024 {
                32
            } else if file_size < 4 * 1024 * 1024 * 1024 {
                48
            } else {
                64
            };
            let raw_chunk = file_size.div_ceil(target_depth.max(1));
            let chunk_size = clamp_chunk_size(raw_chunk, MAX_IO_URING_CHUNK_SIZE);
            let chunk_count = file_size.div_ceil(chunk_size).max(1);
            let target_inflight = chunk_count.min(target_depth).clamp(1, MAX_IO_URING_DEPTH);
            let wait_for = target_inflight.div_ceil(2).max(1);
            FileIoPlan { chunk_size, chunk_count, target_inflight, wait_for }
        }
    }
}

/// Compute batch-read parameters for the given engine.
///
/// Returns a sensible `RangeBatchPlan` even when `request_count` is zero.
#[must_use]
pub fn range_batch_plan(
    request_count: usize,
    total_bytes: usize,
    engine: EngineKind,
) -> RangeBatchPlan {
    if request_count == 0 {
        return RangeBatchPlan { target_inflight: 1, wait_for: 1, coalesce_window_bytes: 0 };
    }

    let cpus = chunk_budget();
    let avg_bytes = total_bytes.div_ceil(request_count).max(1);
    match engine {
        EngineKind::Sync => {
            let target_inflight = request_count.min(cpus.clamp(1, 8)).max(1);
            RangeBatchPlan {
                target_inflight,
                wait_for: target_inflight,
                coalesce_window_bytes: avg_bytes.clamp(64 * 1024, 4 * 1024 * 1024),
            }
        }
        EngineKind::Tokio => {
            let target_inflight = request_count.min((cpus * 2).clamp(1, 16)).max(1);
            RangeBatchPlan {
                target_inflight,
                wait_for: target_inflight.div_ceil(2).max(1),
                coalesce_window_bytes: avg_bytes.clamp(128 * 1024, 8 * 1024 * 1024),
            }
        }
        EngineKind::IoUring => {
            let depth_hint = if avg_bytes <= 256 * 1024 {
                64
            } else if avg_bytes <= 2 * 1024 * 1024 {
                32
            } else {
                16
            };
            let target_inflight = request_count.min(depth_hint).clamp(1, MAX_IO_URING_DEPTH);
            RangeBatchPlan {
                target_inflight,
                wait_for: target_inflight.div_ceil(2).max(1),
                coalesce_window_bytes: avg_bytes.clamp(256 * 1024, 16 * 1024 * 1024),
            }
        }
    }
}

/// Build a vector of `ChunkPlan` entries covering `file_size` bytes using
/// `Sync` engine parameters.
#[must_use]
pub fn build_chunk_plan(file_size: usize) -> Vec<ChunkPlan> {
    let meta = file_chunk_plan(file_size, EngineKind::Sync);
    let mut plan = Vec::with_capacity(meta.chunk_count);
    for i in 0..meta.chunk_count {
        let start = i * meta.chunk_size;
        if start >= file_size {
            break;
        }
        let end = (start + meta.chunk_size).min(file_size);
        let len = end - start;
        if len == 0 {
            break;
        }
        plan.push(ChunkPlan { offset: start as u64, len });
    }
    plan
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_chunk_plan_zero_for_all_engines() {
        for engine in [EngineKind::Sync, EngineKind::Tokio, EngineKind::IoUring] {
            let p = file_chunk_plan(0, engine);
            assert_eq!(p.chunk_count, 1);
            assert_eq!(p.target_inflight, 1);
            assert_eq!(p.wait_for, 1);
        }
    }

    #[test]
    fn file_chunk_plan_chunk_size_within_bounds() {
        let file_size = 200 * 1024 * 1024;
        for engine in [EngineKind::Sync, EngineKind::Tokio] {
            let p = file_chunk_plan(file_size, engine);
            assert!(p.chunk_size >= MIN_CHUNK_SIZE, "chunk_size below MIN: {}", p.chunk_size);
            assert!(p.chunk_size <= MAX_CHUNK_SIZE, "chunk_size above MAX: {}", p.chunk_size);
        }
        let p = file_chunk_plan(file_size, EngineKind::IoUring);
        assert!(p.chunk_size >= MIN_CHUNK_SIZE);
        assert!(p.chunk_size <= MAX_IO_URING_CHUNK_SIZE);
    }

    #[test]
    fn file_chunk_plan_scales_by_engine() {
        let file_size = 512 * 1024 * 1024;
        let sync = file_chunk_plan(file_size, EngineKind::Sync);
        let tokio = file_chunk_plan(file_size, EngineKind::Tokio);
        // Tokio targets more inflight than Sync for large files
        assert!(tokio.target_inflight >= sync.target_inflight);
    }

    #[test]
    fn range_batch_plan_zero_requests() {
        let p = range_batch_plan(0, 0, EngineKind::Sync);
        assert_eq!(p.target_inflight, 1);
        assert_eq!(p.wait_for, 1);
        assert_eq!(p.coalesce_window_bytes, 0);
    }

    #[test]
    fn range_batch_plan_invariants() {
        for engine in [EngineKind::Sync, EngineKind::Tokio, EngineKind::IoUring] {
            let p = range_batch_plan(16, 16 * 1024 * 1024, engine);
            assert!(p.target_inflight >= 1);
            assert!(p.wait_for >= 1);
            assert!(p.wait_for <= p.target_inflight);
            assert!(p.coalesce_window_bytes > 0);
        }
    }

    #[test]
    fn build_chunk_plan_covers_file() {
        let file_size = 200 * 1024 * 1024;
        let plan = build_chunk_plan(file_size);
        assert!(!plan.is_empty());
        let total: usize = plan.iter().map(|c| c.len).sum();
        assert_eq!(total, file_size);
    }

    #[test]
    fn build_chunk_plan_zero_file() {
        // zero-sized file produces one chunk at offset 0 with len 0
        let plan = build_chunk_plan(0);
        assert!(plan.is_empty());
    }

    #[test]
    fn chunk_budget_at_least_one() {
        assert!(chunk_budget() >= 1);
    }
}
