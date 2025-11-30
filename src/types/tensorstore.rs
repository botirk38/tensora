//! Shared types for `TensorStore` format.

use serde::{Deserialize, Serialize};

/// Index entry for `TensorStore` format.
///
/// This structure represents a single tensor's metadata in a `TensorStore` index.
/// It contains compact binary information about the tensor's location, type, and shape.
///
/// Used by both readers (for parsing index entries) and writers (for creating index entries).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IndexEntry {
    /// Shard identifier (which shard file contains this tensor)
    pub shard_id: u8,

    /// Byte offset within the shard file
    pub offset: u64,

    /// Size of the tensor data in bytes
    pub size: u32,

    /// Data type identifier (compact encoding)
    pub dtype: u8,

    /// Number of dimensions (rank) of the tensor
    pub rank: u8,

    /// Length of the tensor name in bytes
    pub name_len: u16,

    /// Shape of the tensor (up to 8 dimensions, 0-padded)
    pub shape: [u32; 8],

    /// Inline storage for short tensor names (up to 16 bytes)
    pub name_inline: [u8; 16],
}
