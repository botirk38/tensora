//! Shared helpers for ServerlessLLM loaders.

use crate::formats::error::{ReaderError, ReaderResult};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// Target on-disk bytes per ServerlessLLM partition for automatic layout (`ceil(total / 512 MiB)` partitions).
pub const RECOMMENDED_PARTITION_TARGET_BYTES: u64 = 512 * 1024 * 1024;

/// Recommended partition count for a model of `total_bytes`, for automatic conversion defaults.
///
/// Formula: `max(1, ceil(total_bytes / 512 MiB))` with no artificial upper bound (beyond `usize`).
#[must_use]
pub fn recommended_partition_count(total_bytes: u64) -> usize {
    if total_bytes == 0 {
        return 1;
    }
    let n = total_bytes.div_ceil(RECOMMENDED_PARTITION_TARGET_BYTES);
    if n > usize::MAX as u64 {
        usize::MAX
    } else {
        n as usize
    }
}

/// Validate that partition exists and is large enough, using cache when possible.
pub fn validate_partition_size(
    partition_path: &Path,
    partition_id: usize,
    required_size: u64,
    cache: &Mutex<HashMap<usize, u64>>,
) -> ReaderResult<u64> {
    if let Ok(guard) = cache.lock()
        && let Some(&cached_size) = guard.get(&partition_id)
    {
        if cached_size < required_size {
            return Err(ReaderError::PartitionTooSmall {
                path: partition_path.display().to_string(),
                actual: cached_size,
                required: required_size,
            });
        }
        return Ok(cached_size);
    }

    let metadata =
        std::fs::metadata(partition_path).map_err(|_| ReaderError::PartitionNotFound {
            partition_id,
            path: partition_path.display().to_string(),
        })?;

    let actual_size = metadata.len();

    if let Ok(mut guard) = cache.lock() {
        guard.insert(partition_id, actual_size);
    }

    if actual_size < required_size {
        return Err(ReaderError::PartitionTooSmall {
            path: partition_path.display().to_string(),
            actual: actual_size,
            required: required_size,
        });
    }

    Ok(actual_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommended_partition_count_zero_and_small() {
        assert_eq!(recommended_partition_count(0), 1);
        assert_eq!(recommended_partition_count(1), 1);
        assert_eq!(
            recommended_partition_count(RECOMMENDED_PARTITION_TARGET_BYTES - 1),
            1
        );
        assert_eq!(
            recommended_partition_count(RECOMMENDED_PARTITION_TARGET_BYTES),
            1
        );
    }

    #[test]
    fn recommended_partition_count_scales_by_half_gib_steps() {
        assert_eq!(
            recommended_partition_count(RECOMMENDED_PARTITION_TARGET_BYTES + 1),
            2
        );
        assert_eq!(
            recommended_partition_count(2 * RECOMMENDED_PARTITION_TARGET_BYTES),
            2
        );
        assert_eq!(
            recommended_partition_count(2 * RECOMMENDED_PARTITION_TARGET_BYTES + 1),
            3
        );
    }

    #[test]
    fn recommended_partition_count_large() {
        let total = 32 * 1024 * 1024 * 1024u64;
        let count = recommended_partition_count(total);
        assert_eq!(count, 64);
    }

    #[test]
    fn validate_partition_size_ok() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("tensor.data_0");
        std::fs::write(&path, vec![0u8; 1024]).unwrap();
        let cache = Mutex::new(HashMap::new());
        let result = validate_partition_size(&path, 0, 512, &cache);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1024);
    }

    #[test]
    fn validate_partition_size_too_small() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("tensor.data_0");
        std::fs::write(&path, vec![0u8; 100]).unwrap();
        let cache = Mutex::new(HashMap::new());
        let result = validate_partition_size(&path, 0, 200, &cache);
        assert!(matches!(result, Err(ReaderError::PartitionTooSmall { .. })));
    }

    #[test]
    fn validate_partition_size_not_found() {
        let path = std::path::Path::new("/nonexistent/tensor.data_99");
        let cache = Mutex::new(HashMap::new());
        let result = validate_partition_size(path, 99, 100, &cache);
        assert!(matches!(result, Err(ReaderError::PartitionNotFound { .. })));
    }

    #[test]
    fn validate_partition_size_uses_cache() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("tensor.data_0");
        std::fs::write(&path, vec![0u8; 1024]).unwrap();
        let cache = Mutex::new(HashMap::new());

        let r1 = validate_partition_size(&path, 0, 512, &cache).unwrap();
        std::fs::remove_file(&path).unwrap();
        let r2 = validate_partition_size(&path, 0, 512, &cache).unwrap();
        assert_eq!(r1, r2);
    }
}
