//! Request grouping and coalescing for batch I/O.
//!
//! Engines that support batch reads use these helpers to:
//! 1. Group requests by file path ([`group_requests_by_file`]).
//! 2. Merge adjacent requests within a configurable window
//!    ([`coalesce_requests`]).
//!
//! The output of `coalesce_requests` is a flat list of
//! [`CoalescedGroup`]s. Each group carries one read range and the list of
//! original requests whose data lives inside it, expressed as relative offsets.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ============================================================================
// Internal types
// ============================================================================

/// An original request annotated with its position in the caller's slice.
#[derive(Debug, Clone)]
pub struct IndexedRequest {
    /// Index of this request in the caller's original input slice.
    pub idx: usize,
    /// Byte offset within the file.
    pub offset: u64,
    /// Number of bytes requested.
    pub len: usize,
}

/// One original request's position within a [`CoalescedGroup`].
#[derive(Debug, Clone)]
pub struct CoalescedMember {
    /// Original request index.
    pub idx: usize,
    /// Byte offset of this member's data relative to the start of the group.
    pub relative_offset: usize,
    /// Number of bytes for this member.
    pub len: usize,
}

/// A merged I/O group: one contiguous read that satisfies one or more original
/// requests.
#[derive(Debug, Clone)]
pub struct CoalescedGroup {
    /// File to read from.
    pub path: PathBuf,
    /// Byte offset of the group's start within the file.
    pub offset: u64,
    /// Total byte length of the merged read.
    pub len: usize,
    /// The original requests whose data lives inside this group.
    pub members: Vec<CoalescedMember>,
}

// ============================================================================
// Result types
// ============================================================================

/// Intermediate per-request result before ordering is restored.
///
/// `(original_index, backing_arc, logical_offset_within_arc, logical_len)`
pub type BatchResult = (usize, Arc<[u8]>, usize, usize);

/// Final per-request result after ordering has been restored.
///
/// `(backing_arc, logical_offset, logical_len)`
pub type FlattenedResult = (Arc<[u8]>, usize, usize);

// ============================================================================
// Public functions
// ============================================================================

/// Group a flat list of `(path, offset, len)` tuples by file path.
///
/// The returned map preserves insertion order within each file's request list.
pub fn group_requests_by_file(
    requests: &[(impl AsRef<Path>, u64, usize)],
) -> HashMap<PathBuf, Vec<IndexedRequest>> {
    requests.iter().enumerate().fold(HashMap::new(), |mut acc, (idx, (path, offset, len))| {
        acc.entry(path.as_ref().to_path_buf()).or_default().push(IndexedRequest {
            idx,
            offset: *offset,
            len: *len,
        });
        acc
    })
}

/// Coalesce per-file request groups into merged read ranges.
///
/// Requests within the same file that are within `window_bytes` of each other
/// are merged into a single read. Requests are sorted by offset before
/// coalescing.
pub fn coalesce_requests(
    grouped: HashMap<PathBuf, Vec<IndexedRequest>>,
    window_bytes: usize,
) -> Vec<CoalescedGroup> {
    let mut groups = Vec::new();

    for (path, mut requests) in grouped {
        requests.sort_unstable_by_key(|r| r.offset);

        let mut current: Option<CoalescedGroup> = None;
        for req in requests {
            let req_end = req.offset.saturating_add(req.len as u64);

            match &mut current {
                Some(group)
                    if req.offset
                        <= group
                            .offset
                            .saturating_add(group.len as u64)
                            .saturating_add(window_bytes as u64) =>
                {
                    let relative_offset =
                        usize::try_from(req.offset.saturating_sub(group.offset))
                            .unwrap_or(usize::MAX);
                    let required_len = relative_offset.saturating_add(req.len);
                    group.len = group.len.max(required_len);
                    group.members.push(CoalescedMember {
                        idx: req.idx,
                        relative_offset,
                        len: req.len,
                    });
                }
                _ => {
                    if let Some(g) = current.take() {
                        groups.push(g);
                    }
                    current = Some(CoalescedGroup {
                        path: path.clone(),
                        offset: req.offset,
                        len: req.len,
                        members: vec![CoalescedMember { idx: req.idx, relative_offset: 0, len: req.len }],
                    });
                }
            }

            // Extend the group if req_end exceeds the current group end.
            if let Some(group) = &mut current {
                let group_end = group.offset.saturating_add(group.len as u64);
                if req_end > group_end {
                    group.len = usize::try_from(req_end.saturating_sub(group.offset))
                        .unwrap_or(usize::MAX);
                }
            }
        }

        if let Some(g) = current.take() {
            groups.push(g);
        }
    }

    groups
}

/// Sort a list of `BatchResult`s by original index and strip the index,
/// returning results in the caller's original request order.
pub fn flatten_results(results: Vec<Vec<BatchResult>>) -> Vec<FlattenedResult> {
    let mut indexed: Vec<BatchResult> = results.into_iter().flatten().collect();
    indexed.sort_unstable_by_key(|(idx, _, _, _)| *idx);
    indexed.into_iter().map(|(_, arc, off, len)| (arc, off, len)).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn group_by_file_single() {
        let reqs = vec![("a.bin", 0u64, 100usize), ("a.bin", 100, 50)];
        let g = group_requests_by_file(&reqs);
        assert_eq!(g.len(), 1);
        assert_eq!(g[Path::new("a.bin")].len(), 2);
    }

    #[test]
    fn group_by_file_multiple() {
        let reqs = vec![("a.bin", 0u64, 10usize), ("b.bin", 0, 20), ("a.bin", 10, 10)];
        let g = group_requests_by_file(&reqs);
        assert_eq!(g.len(), 2);
        assert_eq!(g[Path::new("a.bin")].len(), 2);
        assert_eq!(g[Path::new("b.bin")].len(), 1);
    }

    #[test]
    fn group_by_file_empty() {
        let reqs: Vec<(&str, u64, usize)> = vec![];
        assert!(group_requests_by_file(&reqs).is_empty());
    }

    #[test]
    fn coalesce_merges_adjacent() {
        let reqs = vec![("f", 0u64, 100usize), ("f", 120, 80), ("f", 1024, 64)];
        let groups = coalesce_requests(group_requests_by_file(&reqs), 64);
        // gap of 20 bytes <= window 64, so first two merge; third is separate
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn coalesce_zero_window_touching_still_merges() {
        // touching (gap == 0) should always merge
        let reqs = vec![("f", 0u64, 100usize), ("f", 100, 50)];
        let groups = coalesce_requests(group_requests_by_file(&reqs), 0);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len, 150);
    }

    #[test]
    fn coalesce_zero_window_with_gap_does_not_merge() {
        let reqs = vec![("f", 0u64, 100usize), ("f", 101, 50)];
        let groups = coalesce_requests(group_requests_by_file(&reqs), 0);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn coalesce_cross_file_never_merged() {
        let reqs = vec![("a", 0u64, 100usize), ("b", 0, 100)];
        let groups = coalesce_requests(group_requests_by_file(&reqs), usize::MAX);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn flatten_results_restores_order() {
        let results: Vec<Vec<BatchResult>> = vec![
            vec![(2, Arc::new([3u8]), 0, 1), (0, Arc::new([1u8]), 0, 1)],
            vec![(1, Arc::new([2u8]), 0, 1)],
        ];
        let flat = flatten_results(results);
        assert_eq!(flat.len(), 3);
        assert_eq!(flat[0].0.as_ref(), &[1u8]);
        assert_eq!(flat[1].0.as_ref(), &[2u8]);
        assert_eq!(flat[2].0.as_ref(), &[3u8]);
    }

    #[test]
    fn flatten_results_empty() {
        assert!(flatten_results(vec![]).is_empty());
    }
}
