//! ServerlessLLM format writer.
//!
//! This module provides functionality to write ServerlessLLM tensor index files
//! and partition binary data files.
//!
//! # Format Structure
//!
//! ```text
//! tensor_index.json:
//! {
//!   "tensor_name": [offset, size, [shape...], [stride...], "dtype"],
//!   ...
//! }
//!
//! tensor.data_0: Binary tensor data (partition 0)
//! tensor.data_1: Binary tensor data (partition 1)
//! ...
//! ```
//!
//! # Example Usage (Future)
//!
//! ```rust,ignore
//! use tensor_store::writers::serverlessllm::ServerlessLlmWriter;
//!
//! let writer = ServerlessLlmWriter::new();
//! writer.write_index("tensor_index.json", &tensors).await?;
//! writer.write_partition("tensor.data_0", 0, &data).await?;
//! ```

use crate::writers::IoResult;
use serde::Serialize;
use std::collections::HashMap;

/// Tensor entry for ServerlessLLM index
#[derive(Debug, Serialize)]
pub struct TensorEntry {
    /// Byte offset in partition file
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Tensor strides
    pub stride: Vec<i64>,
    /// Data type string
    pub dtype: String,
}

/// High-level writer for the ServerlessLLM checkpoint format.
#[derive(Debug, Default, Clone, Copy)]
pub struct ServerlessLlmWriter;

impl ServerlessLlmWriter {
    /// Create a new writer instance.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Write `tensor_index.json`.
    pub async fn write_index(
        &self,
        output_path: &str,
        tensors: &HashMap<String, TensorEntry>,
    ) -> IoResult<()> {
        write_index(output_path, tensors).await
    }

    /// Write a partition file (`tensor.data_N`).
    pub async fn write_partition(
        &self,
        output_path: &str,
        partition_id: usize,
        data: &[u8],
    ) -> IoResult<()> {
        write_partition(output_path, partition_id, data).await
    }
}

/// Write tensor_index.json
pub async fn write_index(
    _output_path: &str,
    _tensors: &HashMap<String, TensorEntry>,
) -> IoResult<()> {
    todo!("Implement ServerlessLLM index writing")
}

/// Write partition file (tensor.data_N)
pub async fn write_partition(
    _output_path: &str,
    _partition_id: usize,
    _data: &[u8],
) -> IoResult<()> {
    todo!("Implement ServerlessLLM partition writing")
}
