//! Shared types for `ServerlessLLM` tensor storage format.

use serde::{Deserialize, Serialize};

/// Tensor entry metadata for `ServerlessLLM` index format.
///
/// This structure represents the metadata for a single tensor in a `ServerlessLLM` index file.
/// It contains information about the tensor's location, size, shape, and data type.
///
/// Used by both readers (for parsing index files) and writers (for creating index files).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct TensorEntry {
    /// Byte offset of the tensor data within the partition file
    pub offset: u64,

    /// Size of the tensor data in bytes
    pub size: u64,

    /// Shape of the tensor (dimensions)
    pub shape: Vec<i64>,

    /// Stride information for the tensor
    pub stride: Vec<i64>,

    /// Data type of the tensor (e.g., "float32", "int64")
    pub dtype: String,

    /// Partition ID where this tensor is stored
    pub partition_id: usize,
}
