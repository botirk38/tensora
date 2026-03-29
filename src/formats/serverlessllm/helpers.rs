//! Shared helpers for ServerlessLLM loaders.

use crate::formats::error::{ReaderError, ReaderResult};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

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

    let metadata = std::fs::metadata(partition_path).map_err(|_| ReaderError::PartitionNotFound {
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
