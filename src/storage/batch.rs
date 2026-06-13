//! Batch I/O planning via [`Batcher`].
//!
//! [`Batcher`] turns a [`BatchReadRequest`] into a [`BatchPlan`] — a list of
//! [`CoalescedRead`]s, each representing one merged I/O that satisfies one or
//! more of the original requests.
//!
//! `Batcher` describes *policy* only (how aggressively to merge adjacent
//! requests). It has no knowledge of any specific engine. Engines hold a
//! `Batcher` as a field and choose their own window size.
//!
//! # Constructors
//!
//! [`Batcher::new(n)`] is the only constructor. Pass `0` to merge only
//! touching requests; pass `usize::MAX` to always merge within a file.
//!
//! # Example
//!
//! ```rust,ignore
//! use tensora::storage::batch::Batcher;
//!
//! // Merge adjacent requests separated by up to 256 KiB.
//! let batcher = Batcher::new(256 * 1024);
//! let plan = batcher.plan(&req);
//! for group in &plan.groups {
//!     // issue one I/O for group.len bytes at group.offset in group.path,
//!     // then call group.member_data(member, &raw_bytes) for each member.
//! }
//! ```

use std::collections::BTreeMap;
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

impl CoalescedMember {
    /// Slice this member's data out of the raw bytes returned for its
    /// [`CoalescedRead`].
    ///
    /// Returns `None` if the member's range is out of bounds for `raw`,
    /// which would indicate a corrupt plan or wrong buffer.
    #[inline]
    pub fn data<'a>(&self, raw: &'a [u8]) -> Option<&'a [u8]> {
        let end = self.relative_offset.checked_add(self.len)?;
        raw.get(self.relative_offset..end)
    }
}

/// A single merged read that satisfies one or more original requests.
///
/// Members are in offset order within the file.
#[derive(Debug, Clone)]
pub struct CoalescedRead {
    /// File to read.
    pub path: PathBuf,
    /// Byte offset of the merged read within the file.
    pub offset: u64,
    /// Total byte length of the merged read.
    pub len: usize,
    /// Original requests whose data lives inside this read, in offset order.
    pub members: Vec<CoalescedMember>,
}

/// The result of [`Batcher::plan`].
///
/// Groups are in `(path, offset)` order — deterministic across runs for the
/// same input. Members within each group are in offset order.
#[derive(Debug)]
pub struct BatchPlan {
    /// Merged reads to issue.
    pub groups: Vec<CoalescedRead>,
    /// Number of original requests covered by this plan (== input batch size).
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
/// Adjacent requests within the same file are merged into a single read when
/// the gap between them is at most `coalesce_window_bytes`. A gap of exactly
/// zero (touching) is always merged regardless of the window.
///
/// `Batcher` is engine-agnostic. Each engine chooses its own window based on
/// its I/O characteristics and holds the configured `Batcher` as a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Batcher {
    /// Adjacent requests separated by at most this many bytes are merged.
    ///
    /// `0` means only touching requests are merged. `usize::MAX` means all
    /// requests in the same file are always merged.
    pub coalesce_window_bytes: usize,
}

impl Batcher {
    /// Create a batcher that merges gaps up to `coalesce_window_bytes`.
    ///
    /// `0` means only touching (zero-gap) requests are merged.
    /// `usize::MAX` means all requests in the same file are always merged.
    #[inline]
    #[must_use]
    pub const fn new(coalesce_window_bytes: usize) -> Self {
        Self { coalesce_window_bytes }
    }

    /// Produce a [`BatchPlan`] from a batch request.
    ///
    /// Requests are grouped by file path, sorted by offset within each file,
    /// and merged when the gap between consecutive requests is at most
    /// `self.coalesce_window_bytes`. The returned groups are in `(path, offset)`
    /// order — deterministic for the same input.
    #[must_use]
    pub fn plan<'a>(&self, req: &BatchReadRequest<'a>) -> BatchPlan {
        let request_count = req.len();
        if request_count == 0 {
            return BatchPlan { groups: Vec::new(), request_count: 0 };
        }

        // BTreeMap gives deterministic iteration order (sorted by path).
        let mut by_file: BTreeMap<&Path, Vec<(usize, u64, usize)>> = BTreeMap::new();
        for (i, (path, range)) in req.paths.iter().zip(req.ranges.iter()).enumerate() {
            by_file.entry(path).or_default().push((i, range.offset, range.len));
        }

        let mut groups: Vec<CoalescedRead> = Vec::new();

        for (path, mut entries) in by_file {
            // Sort by offset so we scan left-to-right.
            entries.sort_unstable_by_key(|&(_, offset, _)| offset);

            // Clone path once per file, not once per group.
            let path_buf = path.to_path_buf();
            let mut current: Option<CoalescedRead> = None;

            for (request_index, offset, len) in entries {
                // End of this request as a u64 file offset.
                let req_end: u64 = offset.saturating_add(len as u64);

                match &mut current {
                    Some(g) if should_merge(g, offset, self.coalesce_window_bytes) => {
                        // This request falls within the current group's merge window.
                        // relative_offset: how far into g's buffer this member starts.
                        // Because entries are sorted and should_merge passed, offset >= g.offset.
                        let relative_offset = (offset - g.offset) as usize;
                        let required_len = relative_offset.saturating_add(len);
                        // Extend the group if this request reaches past the current end.
                        if required_len > g.len {
                            g.len = required_len;
                        }
                        g.members.push(CoalescedMember { request_index, relative_offset, len });
                    }
                    _ => {
                        if let Some(g) = current.take() {
                            groups.push(g);
                        }
                        current = Some(CoalescedRead {
                            path: path_buf.clone(),
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

                // Extend the current group if req_end reaches past it.
                // This handles the case where the request starts inside the group
                // but extends beyond it (e.g. a large overlapping range).
                if let Some(g) = &mut current {
                    let group_end: u64 = g.offset.saturating_add(g.len as u64);
                    if req_end > group_end {
                        // Safe: req_end > g.offset always holds here because
                        // req_end >= offset >= g.offset.
                        g.len = (req_end - g.offset) as usize;
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
// Internal helpers
// ============================================================================

/// Returns `true` if `offset` falls within `g`'s current extent plus the
/// coalesce window.
///
/// Precondition: entries are sorted by offset, so `offset >= g.offset`.
/// Using plain arithmetic (not saturating) is safe here because:
/// - `g.offset <= offset` (sort invariant)
/// - `g.len` is at most the file size (bounded by actual I/O)
/// - `window` is caller-controlled; overflow via `window = usize::MAX` is
///   handled by the `u64` cast path: `(g.len as u64).saturating_add(window as u64)`
///   saturates to `u64::MAX`, making the condition always true, which is the
///   intended "unlimited" semantics.
#[inline]
fn should_merge(g: &CoalescedRead, offset: u64, window: usize) -> bool {
    let group_end: u64 = g.offset.saturating_add(g.len as u64);
    let merge_limit: u64 = group_end.saturating_add(window as u64);
    offset <= merge_limit
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{BatchRange, BatchReadRequest};
    use std::path::Path;

    fn make_req<'a>(paths: &'a [&'a Path], ranges: &'a [BatchRange]) -> BatchReadRequest<'a> {
        BatchReadRequest::new(paths, ranges)
    }

    // --- empty / trivial ---

    #[test]
    fn empty_request_produces_empty_plan() {
        let plan = Batcher::new(0).plan(&make_req(&[], &[]));
        assert!(plan.is_empty());
        assert_eq!(plan.request_count, 0);
    }

    #[test]
    fn single_request_one_group() {
        let p = Path::new("/tmp/a.bin");
        let plan = Batcher::new(0).plan(&make_req(&[p], &[BatchRange::new(0, 100)]));
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].offset, 0);
        assert_eq!(plan.groups[0].len, 100);
        assert_eq!(plan.groups[0].members[0].request_index, 0);
    }

    // --- merging ---

    #[test]
    fn touching_requests_always_merge() {
        let p = Path::new("/tmp/b.bin");
        let paths = [p, p];
        let ranges = [BatchRange::new(0, 100), BatchRange::new(100, 50)];
        let plan = Batcher::new(0).plan(&make_req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].len, 150);
    }

    #[test]
    fn gap_within_window_merges() {
        let p = Path::new("/tmp/c.bin");
        let paths = [p, p];
        // gap of 50 bytes, window = 64
        let ranges = [BatchRange::new(0, 100), BatchRange::new(150, 50)];
        let plan = Batcher::new(64).plan(&make_req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].offset, 0);
        assert_eq!(plan.groups[0].len, 200);
    }

    #[test]
    fn gap_beyond_window_stays_separate() {
        let p = Path::new("/tmp/d.bin");
        let paths = [p, p];
        // gap of 101 bytes, window = 100
        let ranges = [BatchRange::new(0, 100), BatchRange::new(201, 50)];
        let plan = Batcher::new(100).plan(&make_req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 2);
    }

    #[test]
    fn overlapping_requests_extend_group() {
        // [0..100] then [50..150]: second starts inside first, extends beyond it.
        let p = Path::new("/tmp/overlap.bin");
        let paths = [p, p];
        let ranges = [BatchRange::new(0, 100), BatchRange::new(50, 100)];
        let plan = Batcher::new(0).plan(&make_req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].len, 150);
    }

    // --- file separation ---

    #[test]
    fn different_files_never_merge() {
        let a = Path::new("/tmp/a.bin");
        let b = Path::new("/tmp/b.bin");
        let paths = [a, b];
        let ranges = [BatchRange::new(0, 100), BatchRange::new(0, 100)];
        let plan = Batcher::new(usize::MAX).plan(&make_req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 2);
    }

    // --- determinism ---

    #[test]
    fn output_order_is_deterministic() {
        // Same input twice must produce identical group order.
        let a = Path::new("/tmp/a.bin");
        let b = Path::new("/tmp/b.bin");
        let paths = [a, b, a];
        let ranges = [BatchRange::new(0, 10), BatchRange::new(0, 10), BatchRange::new(20, 10)];
        let batcher = Batcher::new(0);
        let p1 = batcher.plan(&make_req(&paths, &ranges));
        let p2 = batcher.plan(&make_req(&paths, &ranges));
        let paths1: Vec<_> = p1.groups.iter().map(|g| &g.path).collect();
        let paths2: Vec<_> = p2.groups.iter().map(|g| &g.path).collect();
        assert_eq!(paths1, paths2);
    }

    // --- member data ---

    #[test]
    fn member_data_resolves_correctly() {
        let p = Path::new("/tmp/e.bin");
        let paths = [p, p];
        let ranges = [BatchRange::new(0, 4), BatchRange::new(4, 4)];
        let plan = Batcher::new(0).plan(&make_req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 1);

        let raw = &[10u8, 20, 30, 40, 50, 60, 70, 80];
        let g = &plan.groups[0];
        let m0 = g.members.iter().find(|m| m.request_index == 0).unwrap();
        let m1 = g.members.iter().find(|m| m.request_index == 1).unwrap();
        assert_eq!(m0.data(raw), Some(&[10u8, 20, 30, 40][..]));
        assert_eq!(m1.data(raw), Some(&[50u8, 60, 70, 80][..]));
    }

    #[test]
    fn member_data_out_of_bounds_returns_none() {
        let m = CoalescedMember { request_index: 0, relative_offset: 10, len: 5 };
        assert_eq!(m.data(&[0u8; 4]), None); // 10+5 > 4
    }

    // --- request_count ---

    #[test]
    fn request_count_matches_input() {
        let p = Path::new("/tmp/f.bin");
        let paths = [p, p, p];
        let ranges = [BatchRange::new(0, 10), BatchRange::new(20, 10), BatchRange::new(40, 10)];
        let plan = Batcher::new(0).plan(&make_req(&paths, &ranges));
        assert_eq!(plan.request_count, 3);
    }

    // --- window edge cases ---

    #[test]
    fn zero_window_only_merges_touching() {
        let p = Path::new("/tmp/x.bin");
        let paths = [p, p];
        // gap of 1 byte — should NOT merge
        let ranges = [BatchRange::new(0, 99), BatchRange::new(100, 50)];
        let plan = Batcher::new(0).plan(&make_req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 2);
    }

    #[test]
    fn max_window_merges_all_in_same_file() {
        let p = Path::new("/tmp/y.bin");
        let paths = [p, p, p];
        let ranges =
            [BatchRange::new(0, 10), BatchRange::new(9999, 10), BatchRange::new(99999, 10)];
        let plan = Batcher::new(usize::MAX).plan(&make_req(&paths, &ranges));
        assert_eq!(plan.groups.len(), 1);
    }

    #[test]
    fn new_window_is_stored() {
        assert_eq!(Batcher::new(128 * 1024).coalesce_window_bytes, 128 * 1024);
    }
}
