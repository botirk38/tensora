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

#[cfg(test)]
mod tests {
    use super::{flatten_results, group_requests_by_file, BatchResult};
    use std::path::Path;

    #[test]
    fn test_group_requests_by_file_single_file() {
        let requests = vec![
            ("file1.txt", 0u64, 100usize),
            ("file1.txt", 100, 200),
            ("file1.txt", 300, 50),
        ];

        let groups = group_requests_by_file(&requests);

        assert_eq!(groups.len(), 1);
        let file1_requests = groups.get(Path::new("file1.txt")).unwrap();
        assert_eq!(file1_requests.len(), 3);
        assert_eq!(file1_requests[0].idx, 0);
        assert_eq!(file1_requests[0].offset, 0);
        assert_eq!(file1_requests[0].len, 100);
        assert_eq!(file1_requests[1].idx, 1);
        assert_eq!(file1_requests[2].idx, 2);
    }

    #[test]
    fn test_group_requests_by_file_multiple_files() {
        let requests = vec![
            ("file1.txt", 0u64, 100usize),
            ("file2.txt", 50, 150),
            ("file1.txt", 100, 200),
            ("file3.txt", 0, 300),
            ("file2.txt", 200, 100),
        ];

        let groups = group_requests_by_file(&requests);

        assert_eq!(groups.len(), 3);

        let file1 = groups.get(Path::new("file1.txt")).unwrap();
        assert_eq!(file1.len(), 2);
        assert_eq!(file1[0].idx, 0);
        assert_eq!(file1[1].idx, 2);

        let file2 = groups.get(Path::new("file2.txt")).unwrap();
        assert_eq!(file2.len(), 2);
        assert_eq!(file2[0].idx, 1);
        assert_eq!(file2[1].idx, 4);

        let file3 = groups.get(Path::new("file3.txt")).unwrap();
        assert_eq!(file3.len(), 1);
        assert_eq!(file3[0].idx, 3);
    }

    #[test]
    fn test_group_requests_by_file_empty() {
        let requests: Vec<(&str, u64, usize)> = vec![];
        let groups = group_requests_by_file(&requests);
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn test_flatten_results_all_success() {
        let results = vec![
            vec![(0, vec![1, 2, 3], 0, 3), (2, vec![7, 8, 9], 200, 3)],
            vec![(1, vec![4, 5, 6], 100, 3)],
        ];

        let flattened = flatten_results(results);

        assert_eq!(flattened.len(), 3);
        assert_eq!(flattened[0], (vec![1, 2, 3], 0, 3));
        assert_eq!(flattened[1], (vec![4, 5, 6], 100, 3));
        assert_eq!(flattened[2], (vec![7, 8, 9], 200, 3));
    }

    #[test]
    fn test_flatten_results_preserves_order() {
        let results = vec![
            vec![(3, vec![40], 300, 1), (1, vec![20], 100, 1)],
            vec![
                (0, vec![10], 0, 1),
                (2, vec![30], 200, 1),
                (4, vec![50], 400, 1),
            ],
        ];

        let flattened = flatten_results(results);

        assert_eq!(flattened.len(), 5);
        assert_eq!(flattened[0].0, vec![10]);
        assert_eq!(flattened[1].0, vec![20]);
        assert_eq!(flattened[2].0, vec![30]);
        assert_eq!(flattened[3].0, vec![40]);
        assert_eq!(flattened[4].0, vec![50]);
    }

    #[test]
    fn test_flatten_results_empty() {
        let results: Vec<Vec<BatchResult>> = vec![];
        let flattened = flatten_results(results);
        assert_eq!(flattened.len(), 0);
    }
}
