//! `ServerlessLLM` format writer.
//!
//! This module provides functionality to write `ServerlessLLM` tensor index files
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
//! # Example Usage
//!
//! ```rust,ignore
//! use tensor_store::serverlessllm::ServerlessLlmWriter;
//!
//! // Async usage
//! let writer = ServerlessLlmWriter::new();
//! writer.write_index("tensor_index.json", &tensors).await?;
//! writer.write_partition("tensor.data_0", data).await?;
//!
//! // Sync usage
//! writer.write_index_sync("tensor_index.json", &tensors)?;
//! writer.write_partition_sync("tensor.data_0", &data)?;
//! ```

use crate::backends;
use crate::formats::error::{WriterError, WriterResult};
use std::collections::HashMap;
use std::path::Path;

// Re-export shared TensorEntry type for backwards compatibility
pub use super::types::TensorEntry;

/// High-level writer for the `ServerlessLLM` checkpoint format.
#[derive(Debug, Default, Clone, Copy)]
pub struct ServerlessLlmWriter;

impl ServerlessLlmWriter {
    /// Create a new writer instance.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Write `tensor_index.json` asynchronously.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path where the index file will be written
    /// * `tensors` - `HashMap` of tensor names to their metadata entries
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or if serialization fails.
    #[inline]
    pub async fn write_index(
        &self,
        output_path: impl AsRef<Path>,
        tensors: &HashMap<String, TensorEntry>,
    ) -> WriterResult<()> {
        write_index(output_path, tensors).await
    }

    /// Write a partition file (`tensor.data_N`) asynchronously.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path where the partition file will be written
    /// * `data` - Binary tensor data to write
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    #[inline]
    pub async fn write_partition(
        &self,
        output_path: impl AsRef<Path>,
        data: impl Into<Vec<u8>>,
    ) -> WriterResult<()> {
        write_partition(output_path, data).await
    }

    /// Write `tensor_index.json` synchronously (blocking).
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path where the index file will be written
    /// * `tensors` - `HashMap` of tensor names to their metadata entries
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or if serialization fails.
    #[inline]
    pub fn write_index_sync(
        &self,
        output_path: impl AsRef<Path>,
        tensors: &HashMap<String, TensorEntry>,
    ) -> WriterResult<()> {
        write_index_sync(output_path, tensors)
    }

    /// Write a partition file synchronously (blocking).
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path where the partition file will be written
    /// * `data` - Binary tensor data to write
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    #[inline]
    pub fn write_partition_sync(
        &self,
        output_path: impl AsRef<Path>,
        data: &[u8],
    ) -> WriterResult<()> {
        write_partition_sync(output_path, data)
    }
}

// Helper function to serialize tensors to JSON format
fn serialize_index<S: ::std::hash::BuildHasher>(
    tensors: &HashMap<String, TensorEntry, S>,
) -> WriterResult<Vec<u8>> {
    if tensors.is_empty() {
        return Err(WriterError::InvalidInput(
            "Cannot write empty tensor index".to_owned(),
        ));
    }

    let mut map = serde_json::Map::with_capacity(tensors.len());
    for (name, entry) in tensors {
        let value = serde_json::json!([
            entry.offset,
            entry.size,
            entry.shape,
            entry.stride,
            entry.dtype,
            entry.partition_id
        ]);
        map.insert(name.clone(), value);
    }

    serde_json::to_vec_pretty(&map).map_err(WriterError::from)
}

// Helper function to ensure parent directory exists (async)
async fn ensure_parent_dir_async(path: &Path) -> WriterResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent).await?;
    }
    Ok(())
}

// Helper function to ensure parent directory exists (sync)
fn ensure_parent_dir_sync(path: &Path) -> WriterResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Write `tensor_index.json` asynchronously.
///
/// # Arguments
///
/// * `output_path` - Path where the index file will be written
/// * `tensors` - `HashMap` of tensor names to their metadata entries
///
/// # Errors
///
/// Returns an error if:
/// - The tensors `HashMap` is empty
/// - Parent directory cannot be created
/// - Serialization fails
/// - File cannot be written
pub async fn write_index(
    output_path: impl AsRef<Path>,
    tensors: &HashMap<String, TensorEntry>,
) -> WriterResult<()> {
    let path = output_path.as_ref();
    ensure_parent_dir_async(path).await?;

    let json = serialize_index(tensors)?;
    backends::async_backend()
        .write_all(path, json)
        .await
        .map_err(WriterError::from)
}

/// Write partition file (`tensor.data_N`) asynchronously.
///
/// # Arguments
///
/// * `output_path` - Path where the partition file will be written
/// * `data` - Binary tensor data to write
///
/// # Errors
///
/// Returns an error if:
/// - Parent directory cannot be created
/// - File cannot be written
#[inline]
pub async fn write_partition(
    output_path: impl AsRef<Path>,
    data: impl Into<Vec<u8>>,
) -> WriterResult<()> {
    let path = output_path.as_ref();
    ensure_parent_dir_async(path).await?;
    let bytes = data.into();
    backends::async_backend()
        .write_all(path, bytes)
        .await
        .map_err(WriterError::from)
}

/// Write `tensor_index.json` synchronously.
///
/// # Arguments
///
/// * `output_path` - Path where the index file will be written
/// * `tensors` - `HashMap` of tensor names to their metadata entries
///
/// # Errors
///
/// Returns an error if:
/// - The tensors `HashMap` is empty
/// - Parent directory cannot be created
/// - Serialization fails
/// - File cannot be written
#[inline]
pub fn write_index_sync<S: ::std::hash::BuildHasher>(
    output_path: impl AsRef<Path>,
    tensors: &HashMap<String, TensorEntry, S>,
) -> WriterResult<()> {
    let path = output_path.as_ref();
    ensure_parent_dir_sync(path)?;

    let json = serialize_index(tensors)?;
    std::fs::write(path, json).map_err(WriterError::from)
}

/// Write partition file synchronously.
///
/// # Arguments
///
/// * `output_path` - Path where the partition file will be written
/// * `data` - Binary tensor data to write
///
/// # Errors
///
/// Returns an error if:
/// - Parent directory cannot be created
/// - File cannot be written
#[inline]
pub fn write_partition_sync(output_path: impl AsRef<Path>, data: &[u8]) -> WriterResult<()> {
    let path = output_path.as_ref();
    ensure_parent_dir_sync(path)?;
    std::fs::write(path, data).map_err(WriterError::from)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tempfile::TempDir;

    fn sample_entries() -> HashMap<String, TensorEntry> {
        HashMap::from([(
            "w".to_owned(),
            TensorEntry {
                offset: 0,
                size: 4,
                shape: vec![2, 2],
                stride: vec![2, 1],
                dtype: "f32".to_owned(),
                partition_id: 1,
            },
        )])
    }

    #[test]
    fn serialize_index_rejects_empty() {
        let map: HashMap<String, TensorEntry> = HashMap::new();
        let err = serialize_index(&map).unwrap_err();
        assert!(matches!(err, WriterError::InvalidInput(_)));
    }

    #[test]
    fn write_index_and_partition_sync() {
        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("nested").join("tensor_index.json");
        let part_path = dir.path().join("tensor.data_1");

        write_index_sync(&index_path, &sample_entries()).expect("write index");
        write_partition_sync(&part_path, b"abcd").expect("write partition");

        let json: Value = serde_json::from_slice(&std::fs::read(index_path).unwrap()).unwrap();
        let entry = json.get("w").expect("tensor entry");
        assert_eq!(entry[0], 0);
        assert_eq!(entry[1], 4);
        assert_eq!(std::fs::read(part_path).unwrap(), b"abcd");
    }

    #[test]
    fn write_index_and_partition_async() {
        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("async").join("tensor_index.json");
        let part_path = dir.path().join("tensor.data_1");

        crate::test_utils::run_async(async {
            write_index(&index_path, &sample_entries())
                .await
                .expect("write index async");
            write_partition(&part_path, b"xyz".to_vec())
                .await
                .expect("write partition async");
        });

        let json: Value = serde_json::from_slice(&std::fs::read(index_path).unwrap()).unwrap();
        assert_eq!(json["w"][5], 1);
        assert_eq!(std::fs::read(part_path).unwrap(), b"xyz");
    }
}
