//! `ServerlessLLM` format serializer.
//!
//! This module provides functionality to write `ServerlessLLM` tensor index files
//! and partition binary data files.
//!
//! # Format Structure
//!
//! ```text
//! tensor_index.json:
//! {
//!   "tensor_name": [offset, size, [shape...], [stride...], "dtype", partition_id],
//!   ...
//! }
//!
//! tensor.data_0: Binary tensor data (partition 0)
//! tensor.data_1: Binary tensor data (partition 1)
//! ...
//! ```

use crate::formats::error::{WriterError, WriterResult};
use crate::formats::traits::{AsyncSerializer, SyncSerializer};
use crate::io::sync::Sync;
use crate::io::tokio::Tokio;
use crate::io::{AsyncIo, BlockingIo};
use std::collections::HashMap;
use std::path::Path;

use super::ids::PartitionId;
use super::index::TensorDescriptor;

/// Entry for writing tensors to index.
#[derive(Debug, Clone)]
pub struct TensorWriteEntry {
    pub offset: u64,
    pub size: u64,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub dtype: String,
    pub partition_id: PartitionId,
}

impl From<&TensorDescriptor> for TensorWriteEntry {
    fn from(desc: &TensorDescriptor) -> Self {
        Self {
            offset: desc.offset,
            size: desc.size as u64,
            shape: desc.shape.to_vec(),
            stride: desc.stride.to_vec(),
            dtype: desc.dtype.to_string(),
            partition_id: desc.partition_id,
        }
    }
}

/// Input data for writing a ServerlessLLM model.
#[derive(Debug, Clone)]
pub struct WriteInput {
    pub index: HashMap<String, TensorWriteEntry>,
    pub partitions: Vec<Vec<u8>>,
}

/// Stateful writer for the ServerlessLLM format.
#[derive(Debug, Default)]
pub struct Writer;

impl Writer {
    pub const fn new() -> Self {
        Self
    }
}

impl AsyncSerializer for Writer {
    type Input = WriteInput;

    async fn write(path: &Path, data: &Self::Input) -> WriterResult<()> {
        write_model_to_dir(path, &data.index, &data.partitions).await
    }
}

impl SyncSerializer for Writer {
    type Input = WriteInput;

    fn write_sync(path: &Path, data: &Self::Input) -> WriterResult<()> {
        write_model_to_dir_sync(path, &data.index, &data.partitions)
    }
}

async fn write_model_to_dir(
    directory: &Path,
    index: &HashMap<String, TensorWriteEntry>,
    partitions: &[Vec<u8>],
) -> WriterResult<()> {
    tokio::fs::create_dir_all(directory).await?;
    let index_path = directory.join("tensor_index.json");
    write_index(&index_path, index).await?;
    for (i, data) in partitions.iter().enumerate() {
        let part_path = directory.join(format!("tensor.data_{}", i));
        write_partition(&part_path, data).await?;
    }
    Ok(())
}

fn write_model_to_dir_sync(
    directory: &Path,
    index: &HashMap<String, TensorWriteEntry>,
    partitions: &[Vec<u8>],
) -> WriterResult<()> {
    std::fs::create_dir_all(directory)?;
    let index_path = directory.join("tensor_index.json");
    write_index_sync(&index_path, index)?;
    for (i, data) in partitions.iter().enumerate() {
        let part_path = directory.join(format!("tensor.data_{}", i));
        write_partition_sync(&part_path, data)?;
    }
    Ok(())
}

pub async fn write_index(
    output_path: impl AsRef<Path>,
    tensors: &HashMap<String, TensorWriteEntry>,
) -> WriterResult<()> {
    let path = output_path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(WriterError::from)?;
    }
    let json = serialize_index(tensors)?;
    let engine = Tokio::new();
    engine
        .write_file(path, &json)
        .await
        .map_err(WriterError::from)?;
    engine.sync_all(path).await.map_err(WriterError::from)
}

pub fn write_index_sync(
    output_path: impl AsRef<Path>,
    tensors: &HashMap<String, TensorWriteEntry>,
) -> WriterResult<()> {
    let path = output_path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(WriterError::from)?;
    }
    let json = serialize_index(tensors)?;
    let engine = Sync::new();
    engine.write_file(path, &json).map_err(WriterError::from)?;
    engine.sync_all(path).map_err(WriterError::from)
}

fn serialize_index(tensors: &HashMap<String, TensorWriteEntry>) -> WriterResult<Vec<u8>> {
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
            entry.partition_id.as_usize()
        ]);
        map.insert(name.clone(), value);
    }
    serde_json::to_vec_pretty(&map).map_err(WriterError::from)
}

pub async fn write_partition(output_path: impl AsRef<Path>, data: &[u8]) -> WriterResult<()> {
    let path = output_path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(WriterError::from)?;
    }
    let engine = Tokio::new();
    engine
        .write_file(path, data)
        .await
        .map_err(WriterError::from)?;
    engine.sync_all(path).await.map_err(WriterError::from)
}

pub fn write_partition_sync(output_path: impl AsRef<Path>, data: &[u8]) -> WriterResult<()> {
    let path = output_path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(WriterError::from)?;
    }
    let engine = Sync::new();
    engine.write_file(path, data).map_err(WriterError::from)?;
    engine.sync_all(path).map_err(WriterError::from)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::TensorWriteEntry;
    use crate::formats::error::WriterError;
    use crate::formats::serverlessllm::ids::PartitionId;
    use crate::formats::serverlessllm::serializer::{
        serialize_index, write_index, write_index_sync, write_partition, write_partition_sync,
    };
    use serde_json::Value;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn sample_entries() -> HashMap<String, TensorWriteEntry> {
        HashMap::from([(
            "w".to_owned(),
            TensorWriteEntry {
                offset: 0,
                size: 4,
                shape: vec![2, 2],
                stride: vec![2, 1],
                dtype: "f32".to_owned(),
                partition_id: PartitionId::new(1),
            },
        )])
    }

    #[test]
    fn serialize_index_rejects_empty() {
        let map: HashMap<String, TensorWriteEntry> = HashMap::new();
        let err = serialize_index(&map).unwrap_err();
        assert!(matches!(err, WriterError::InvalidInput(_)));
    }

    #[test]
    fn write_index_and_partition_sync() {
        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("tensor_index.json");
        let entries = sample_entries();
        write_index_sync(&index_path, &entries).unwrap();
        assert!(index_path.exists());

        let part_path = dir.path().join("tensor.data_0");
        write_partition_sync(&part_path, b"abcd").unwrap();
        assert_eq!(std::fs::read(&part_path).unwrap(), b"abcd");
    }

    #[test]
    fn serialize_index_produces_valid_json() {
        let entries = sample_entries();
        let json = serialize_index(&entries).unwrap();
        let parsed: Value = serde_json::from_slice(&json).unwrap();
        let w = parsed.get("w").unwrap().as_array().unwrap();
        assert_eq!(w[0], 0u64);
        assert_eq!(w[1], 4u64);
        assert_eq!(w[4], "f32");
        assert_eq!(w[5], 1u64);
    }

    #[test]
    fn write_index_async() {
        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("tensor_index.json");
        let entries = sample_entries();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(write_index(&index_path, &entries)).unwrap();
        assert!(index_path.exists());
    }

    #[test]
    fn write_partition_async() {
        let dir = TempDir::new().unwrap();
        let part_path = dir.path().join("tensor.data_0");
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(write_partition(&part_path, b"hello")).unwrap();
        assert_eq!(std::fs::read(&part_path).unwrap(), b"hello");
    }
}
