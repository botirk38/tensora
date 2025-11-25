use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub type BatchResult = (usize, Vec<u8>, usize, usize);
pub type FlattenedResult = (Vec<u8>, usize, usize);

#[derive(Debug, Clone)]
pub struct IndexedRequest {
    pub idx: usize,
    pub offset: u64,
    pub len: usize,
}

/// Groups user requests by file path and annotates them with their input index.
pub fn group_requests_by_file(
    requests: &[(impl AsRef<Path>, u64, usize)],
) -> HashMap<PathBuf, Vec<IndexedRequest>> {
    requests
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut acc, (idx, (path, offset, len))| {
            acc.entry(path.as_ref().to_path_buf())
                .or_default()
                .push(IndexedRequest {
                    idx,
                    offset: *offset,
                    len: *len,
                });
            acc
        })
}

/// Flattens per-file results into the original request order.
pub fn flatten_results(results: Vec<Vec<BatchResult>>) -> Vec<FlattenedResult> {
    let mut indexed: Vec<(usize, Vec<u8>, usize, usize)> = results.into_iter().flatten().collect();
    indexed.sort_unstable_by_key(|(idx, _, _, _)| *idx);
    indexed
        .into_iter()
        .map(|(_, data, offset, len)| (data, offset, len))
        .collect()
}
