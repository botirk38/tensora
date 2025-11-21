//! TensorStore format writer.
//!
//! This module exposes a lightweight [`TensorStoreWriter`] that mirrors the
//! layout produced by the (future) TensorStore reader. All methods are async
//! placeholders so the DX stays consistent while format details are
//! implemented.
//!
//! ```rust,ignore
//! use tensor_store::writers::tensorstore::{TensorStoreWriter, TensorStoreIndexEntry};
//!
//! let writer = TensorStoreWriter::new();
//! let entries = vec![TensorStoreIndexEntry::default()];
//! writer.write_index("model.index", &entries).await?;
//! writer.write_shard("shard_0.bin", 0, &[0u8; 1024]).await?;
//! ```

use crate::writers::IoResult;

/// Entry describing a TensorStore tensor in the index file.
#[derive(Debug, Default, Clone)]
pub struct TensorStoreIndexEntry {
    /// Which shard file (0-255)
    pub shard_id: u8,
    /// Byte offset within shard
    pub offset: u64,
    /// Tensor data size in bytes
    pub size: u32,
    /// Data type identifier
    pub dtype: u8,
    /// Number of dimensions
    pub rank: u8,
    /// Length of tensor name
    pub name_len: u16,
    /// Tensor shape (up to 8 dimensions)
    pub shape: [u32; 8],
    /// Inline tensor name (up to 16 bytes)
    pub name_inline: [u8; 16],
}

/// High-level writer for TensorStore checkpoint artifacts.
#[derive(Debug, Default, Clone, Copy)]
pub struct TensorStoreWriter;

impl TensorStoreWriter {
    /// Create a new writer instance.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Write the TensorStore index file.
    pub async fn write_index(
        &self,
        output_path: &str,
        entries: &[TensorStoreIndexEntry],
    ) -> IoResult<()> {
        write_index(output_path, entries).await
    }

    /// Write a binary shard containing tensor data.
    pub async fn write_shard(&self, output_path: &str, shard_id: u8, data: &[u8]) -> IoResult<()> {
        write_shard(output_path, shard_id, data).await
    }
}

/// Write a TensorStore index file.
pub async fn write_index(_output_path: &str, _entries: &[TensorStoreIndexEntry]) -> IoResult<()> {
    todo!("Implement TensorStore index writing")
}

/// Write a TensorStore shard file.
pub async fn write_shard(_output_path: &str, _shard_id: u8, _data: &[u8]) -> IoResult<()> {
    todo!("Implement TensorStore shard writing")
}
