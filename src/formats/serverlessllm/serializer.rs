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

use crate::backends;
use crate::formats::error::{WriterError, WriterResult};
use crate::formats::traits::{AsyncSerializer, SyncSerializer};
use std::collections::HashMap;
use std::path::Path;

use super::index::TensorDescriptor;

/// Entry for writing tensors to index.
#[derive(Debug, Clone)]
pub struct TensorWriteEntry {
    pub offset: u64,
    pub size: u64,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub dtype: String,
    pub partition_id: usize,
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
    ensure_parent_dir_async(path).await?;
    let json = serialize_index(tensors)?;
    let mut writer = backends::AsyncWriter::create(path)
        .await
        .map_err(WriterError::from)?;
    writer.write_all(&json).await.map_err(WriterError::from)
}

pub fn write_index_sync(
    output_path: impl AsRef<Path>,
    tensors: &HashMap<String, TensorWriteEntry>,
) -> WriterResult<()> {
    let path = output_path.as_ref();
    ensure_parent_dir_sync(path)?;
    let json = serialize_index(tensors)?;
    let mut writer = backends::SyncWriter::create(path)?;
    writer.write_all(&json).map_err(WriterError::from)
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
            entry.partition_id
        ]);
        map.insert(name.clone(), value);
    }
    serde_json::to_vec_pretty(&map).map_err(WriterError::from)
}

pub async fn write_partition(output_path: impl AsRef<Path>, data: &[u8]) -> WriterResult<()> {
    let path = output_path.as_ref();
    ensure_parent_dir_async(path).await?;
    let mut writer = backends::AsyncWriter::create(path).await?;
    writer.write_all(data).await.map_err(WriterError::from)
}

pub fn write_partition_sync(output_path: impl AsRef<Path>, data: &[u8]) -> WriterResult<()> {
    let path = output_path.as_ref();
    ensure_parent_dir_sync(path)?;
    let mut writer = backends::SyncWriter::create(path)?;
    writer.write_all(data).map_err(WriterError::from)
}

async fn ensure_parent_dir_async(path: &Path) -> WriterResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent).await?;
    }
    Ok(())
}

fn ensure_parent_dir_sync(path: &Path) -> WriterResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::TensorWriteEntry;
    use crate::formats::error::WriterError;
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
                partition_id: 1,
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
            write_partition(&part_path, b"xyz")
                .await
                .expect("write partition async");
        });

        let json: Value = serde_json::from_slice(&std::fs::read(index_path).unwrap()).unwrap();
        assert_eq!(json["w"][5], 1);
        assert_eq!(std::fs::read(part_path).unwrap(), b"xyz");
    }
}
