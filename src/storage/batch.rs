//! Batch I/O planning via [`Batcher`].
//!
//! [`Batcher`] is the shared entity that all engines use to turn a
//! [`BatchReadRequest`] into a list of [`CoalescedRead`]s — merged,
//! sorted reads that satisfy one or more of the original requests.
//!
//! # Why a struct, not free functions?
//!
//! The coalescing window is an engine-level tuning knob. Different engines
//! (sync, tokio, io_uring) want different windows; wrapping the parameter in a
//! struct lets each engine hold a pre-configured `Batcher` and pass it to
//! `plan()` without repeating the window at every call site.
//!
//! # Example
//!
//! ```rust,ignore
//! use tensora::storage::batch::{Batcher, BatchPlan};
//! use tensora::storage::BatchReadRequest;
//!
//! let batcher = Batcher::new(512 * 1024); // 512 KiB coalesce window
//! let plan: BatchPlan = batcher.plan(&req);
//! for read in &plan.groups {
//!     // issue one I/O for read.len bytes at read.offset in read.path
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::BatchReadRequest;

// ============================================================================
// Output types
// ============================================================================

/// One original request's position inside a [`CoalescedRead`].
#[derive(Debug, Clone)]
pub struct CoalescedMember {
    /// Index of this request in the original [`BatchReadRequest`].
    pub request_index: usize,
    /// Byte offset of this member's data relative to the start of the merged read.
    pub relative_offset: usize,
    /// Number of bytes for this member.
    pub len: usize,
}

/// A single merged read that satisfies one or more original requests.
#[derive(Debug, Clone)]
pub struct CoalescedRead {
    /// File to read.
    pub path: PathBuf,
    /// Byte offset of the merged read within the file.
    pub offset: u64,
    /// Total byte length of the merged read.
    pub len: usize,
    /// The original requests whose data lives inside this read.
    pub members: Vec<CoalescedMember>,
}

impl CoalescedRead {
    /// Resolve a member's data from the raw bytes returned for this read.
    ///
    /// Returns `None` if the member's range is out of bounds for `raw`.
    #[inline]
    pub fn member_slice<'a>(&self, member: &CoalescedMember, raw: &'a [u8]) -> Option<&'a [u8]> {
        let start = member.relative_offset;
        let end = start.checked_add(member.len)?;
        raw.get(start..end)
    }
}

/// The result of [`Batcher::plan`]: a set of merged reads in file-offset order.
#[derive(Debug)]
pub struct BatchPlan {
    /// Merged reads to issue, in arbitrary order across files.
    pub groups: Vec<CoalescedRead>,
    /// Number of original requests covered by this plan (always == input batch size).
    pub request_count: usize,
}

impl BatchPlan {
    /// Returns `true` if there are no reads to issue.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

// ============================================================================
// Batcher
// ============================================================================

/// Plans a [`BatchReadRequest`] into coalesced reads.
///
/// Adjacent requests within the same file that are within
/// `coalesce_window_bytes` of each other are merged into a single read.
/// Setting the window to `0` still merges requests that are exactly touching.
///
/// # Constructors
///
/// | Method | Window |
/// |---|---|
/// | [`Batcher::new`] | caller-specified |
/// | [`Batcher::for_sync`] | 512 KiB (good default for O_DIRECT / buffered I/O) |
/// | [`Batcher::for_tokio`] | 1 MiB (async I/O benefits from larger merges) |
/// | [`Batcher::for_io_uring`] | 2 MiB (multi-worker ring can amortise overhead) |
#[derive(Debug, Clone, Copy)]
pub struct Batcher {
    /// Requests within the same file separated by at most this many bytes are merged.
    pub coalesce_window_bytes: usize,
}

impl Batcher {
    /// Create a batcher with an explicit coalesce window.
    #[inline]
    #[must_use]
    pub const fn new(coalesce_window_bytes: usize) -> Self {
        Self { coalesce_window_bytes }
    }

    /// Tuned for synchronous O_DIRECT / buffered I/O.
    #[inline]
    #[must_use]
    pub const fn for_sync() -> Self {
        Self::new(512 * 1024)
    }

    /// Tuned for Tokio async I/O.
    #[inline]
    #[must_use]
    pub const fn for_tokio() -> Self {
        Self::new(1024 * 1024)
    }

    /// Tuned for Linux io_uring multi-worker I/O.
    #[inline]
    #[must_use]
    pub const fn for_io_uring() -> Self {
        Self::new(2 * 1024 * 1024)
    }

    /// Produce a [`BatchPlan`] from a batch request.
    ///
    /// Requests are grouped by file path, sorted by offset within each file,
    /// and merged when the gap between consecutive requests is at most
    /// `self.coalesce_window_bytes`.
    #[must_use]
    pub fn plan<'a>(&self, req: &BatchReadRequest<'a>) -> BatchPlan {
        let request_count = req.len();
        if request_count == 0 {
            return BatchPlan { groups: Vec::new(), request_count: 0 };
        }

        // Group by file: (path_index, offset, len, original_request_index)
        let mut by_file: HashMap<&Path, Vec<(usize, u64, usize)>> = HashMap::new();
        for (i, (path, range)) in req.paths.iter().zip(req.ranges.iter()).enumerate() {
            by_file.entry(path).or_default().push((i, range.offset, range.len));
        }

        let mut groups: Vec<CoalescedRead> = Vec::new();

        for (path, mut entries) in by_file {
            // Sort by offset so we scan left-to-right.
            entries.sort_unstable_by_key(|&(_, offset, _)| offset);

            let mut current: Option<CoalescedRead> = None;

            for (request_index, offset, len) in entries {
                let req_end = offset.saturating_add(len as u64);

                match &mut current {
                    Some(g)
                        if offset
                            <= g.offset
                                .saturating_add(g.len as u64)
                                .saturating_add(self.coalesce_window_bytes as u64) =>
                    {
                        // Merge into the current group.
                        let relative_offset =
                            usize::try_from(offset.saturating_sub(g.offset)).unwrap_or(usize::MAX);
                        let required_len = relative_offset.saturating_add(len);
                        g.len = g.len.max(required_len);
                        g.members.push(CoalescedMember { request_index, relative_offset, len });
                    }
                    _ => {
                        if let Some(g) = current.take() {
                            groups.push(g);
                        }
                        current = Some(CoalescedRead {
                            path: path.to_path_buf(),
                            offset,
                            len,
                            members: vec![CoalescedMember {
                                request_index,
                                relative_offset: 0,
                                len,
                            }],
                        });
                    }
                }

                // Extend the current group if this request reaches further.
                if let Some(g) = &mut current {
                    let group_end = g.offset.saturating_add(g.len as u64);
                    if req_end > group_end {
                        g.len = usize::try_from(req_end.saturating_sub(g.offset))
                            .unwrap_or(usize::MAX);
                    }
                }
            }

            if let Some(g) = current.take() {
                groups.push(g);
            }
        }

        BatchPlan { groups, request_count }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{BatchRange, BatchReadRequest};
    use std::path::Path;

    fn req<'a>(
        paths: &'a [&'a Path],
        ranges: &'a [BatchRange],
    ) -> BatchReadRequest<'a> {
        BatchReadRequest::new(paths, ranges)
    }

    #[test]
    fn empty_request_produces_empty_plan() {
        let batcher = Batcher::new(0);
        let plan = batcher.plan(&req(&[], &[]));
        assert!(plan.is_empty());
        assert_eq!(plan.request_count, 0);
    }

    #[test]
    fn single_request_one_group() {
        let p = Path::new("/tmp/a.bin");
        let paths = [p];
        let ranges = [BatchRange::new(0, 100)];
        let batcher = Batcher::new(0);
        let plan = batcher.plan(&req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].offset, 0);
        assert_eq!(plan.groups[0].len, 100);
        assert_eq!(plan.groups[0].members[0].request_index, 0);
    }

    #[test]
    fn adjacent_requests_merge_with_zero_window() {
        let p = Path::new("/tmp/b.bin");
        let paths = [p, p];
        let ranges = [BatchRange::new(0, 100), BatchRange::new(100, 50)];
        let batcher = Batcher::new(0);
        let plan = batcher.plan(&req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 1, "touching requests should merge");
        assert_eq!(plan.groups[0].len, 150);
    }

    #[test]
    fn gap_within_window_merges() {
        let p = Path::new("/tmp/c.bin");
        let paths = [p, p];
        let ranges = [BatchRange::new(0, 100), BatchRange::new(150, 50)];
        let batcher = Batcher::new(64); // 50-byte gap < 64-byte window
        let plan = batcher.plan(&req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].len, 200); // 0..200 covers both
    }

    #[test]
    fn gap_beyond_window_stays_separate() {
        let p = Path::new("/tmp/d.bin");
        let paths = [p, p];
        let ranges = [BatchRange::new(0, 100), BatchRange::new(201, 50)];
        let batcher = Batcher::new(100); // 101-byte gap > 100-byte window
        let plan = batcher.plan(&req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 2);
    }

    #[test]
    fn different_files_never_merge() {
        let a = Path::new("/tmp/a.bin");
        let b = Path::new("/tmp/b.bin");
        let paths = [a, b];
        let ranges = [BatchRange::new(0, 100), BatchRange::new(0, 100)];
        let batcher = Batcher::new(usize::MAX);
        let plan = batcher.plan(&req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 2);
    }

    #[test]
    fn member_slice_resolves_correctly() {
        let p = Path::new("/tmp/e.bin");
        let paths = [p, p];
        let ranges = [BatchRange::new(0, 4), BatchRange::new(4, 4)];
        let batcher = Batcher::new(0);
        let plan = batcher.plan(&req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 1);
        let raw = &[10u8, 20, 30, 40, 50, 60, 70, 80];
        let g = &plan.groups[0];
        // members are sorted by offset; member 0 = bytes 0..4, member 1 = 4..8
        let m0 = g.members.iter().find(|m| m.request_index == 0).unwrap();
        let m1 = g.members.iter().find(|m| m.request_index == 1).unwrap();
        assert_eq!(g.member_slice(m0, raw), Some(&[10u8, 20, 30, 40][..]));
        assert_eq!(g.member_slice(m1, raw), Some(&[50u8, 60, 70, 80][..]));
    }

    #[test]
    fn request_count_matches_input() {
        let p = Path::new("/tmp/f.bin");
        let paths = [p, p, p];
        let ranges = [BatchRange::new(0, 10), BatchRange::new(20, 10), BatchRange::new(40, 10)];
        let batcher = Batcher::new(0);
        let plan = batcher.plan(&req(&paths, &ranges));
        assert_eq!(plan.request_count, 3);
    }

    #[test]
    fn preset_constructors_are_ordered_by_window() {
        assert!(Batcher::for_sync().coalesce_window_bytes <= Batcher::for_tokio().coalesce_window_bytes);
        assert!(Batcher::for_tokio().coalesce_window_bytes <= Batcher::for_io_uring().coalesce_window_bytes);
    }
}
