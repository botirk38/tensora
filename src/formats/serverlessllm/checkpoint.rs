//! `ServerlessLLM` format checkpoint.
//!
//! This module provides a format-specific checkpoint type that loads and saves
//! ServerlessLLM models. Loading produces a [`Model`]; saving writes the
//! `tensor_index.json` metadata file and the partition data files
//! (`tensor.data_0`, `tensor.data_1`, ...).
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

use crate::formats::error::{LoadError, LoadResult, SaveError, SaveResult};
use crate::formats::tensor::Dtype;
use crate::formats::traits::Checkpoint as CheckpointTrait;
use crate::formats::{AsyncBackend, Backend};
use crate::io::mmap::Mmap;
use crate::io::sync::Sync;
use crate::io::tokio::Tokio;
use crate::io::{AsyncIo, BlockingIo, MmapIo};
use std::collections::HashMap;


use std::path::Path;

use super::ids::PartitionId;
use super::index::{Index, TensorDescriptor};
use super::model::Model;

// ============================================================================
// TensorWriteEntry
// ============================================================================

/// Entry for writing a tensor into the ServerlessLLM index.
#[derive(Debug, Clone)]
pub struct TensorWriteEntry {
    offset: u64,
    size: u64,
    shape: Vec<usize>,
    stride: Vec<usize>,
    dtype: Dtype,
    partition_id: PartitionId,
}

impl TensorWriteEntry {
    /// Create a new validated tensor write entry.
    ///
    /// # Errors
    ///
    /// Returns `SaveError::InvalidInput` if:
    /// - `offset + size` would overflow
    /// - `shape.len() != stride.len()`
    /// - `dtype` is `Dtype::Unknown`
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        offset: u64,
        size: u64,
        shape: impl Into<Vec<usize>>,
        stride: impl Into<Vec<usize>>,
        dtype: Dtype,
        partition_id: PartitionId,
    ) -> SaveResult<Self> {
        if dtype == Dtype::Unknown {
            return Err(SaveError::InvalidInput(
                "cannot use Unknown dtype for tensor".to_owned(),
            ));
        }

        let shape = shape.into();
        let stride = stride.into();

        if shape.len() != stride.len() {
            return Err(SaveError::InvalidInput(format!(
                "shape length ({}) must match stride length ({})",
                shape.len(),
                stride.len()
            )));
        }

        // Validate offset + size doesn't overflow
        let _ = offset
            .checked_add(size)
            .ok_or_else(|| SaveError::InvalidInput("offset + size overflow".to_owned()))?;

        Ok(Self {
            offset,
            size,
            shape,
            stride,
            dtype,
            partition_id,
        })
    }

    /// Returns the byte offset within the partition.
    #[inline]
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the tensor size in bytes.
    #[inline]
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the tensor shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the tensor stride.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// Returns the dtype.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Returns the partition ID.
    #[inline]
    #[must_use]
    pub fn partition_id(&self) -> PartitionId {
        self.partition_id
    }
}

impl From<&TensorDescriptor> for TensorWriteEntry {
    fn from(desc: &TensorDescriptor) -> Self {
        Self {
            offset: desc.offset(),
            size: desc.size() as u64,
            shape: desc.shape().to_vec(),
            stride: desc.stride().to_vec(),
            dtype: desc.dtype(),
            partition_id: desc.partition_id(),
        }
    }
}

// ============================================================================
// Checkpoint
// ============================================================================

/// A ServerlessLLM checkpoint, ready to serialize.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    index: HashMap<String, TensorWriteEntry>,
    partitions: Vec<Vec<u8>>,
}

impl Checkpoint {
    /// Create a new validated checkpoint.
    ///
    /// # Errors
    ///
    /// Returns `SaveError::InvalidInput` if:
    /// - index is empty
    /// - a tensor name is empty
    /// - a tensor references a partition that doesn't exist
    /// - offset + size would overflow within a partition
    pub fn new(
        index: impl IntoIterator<Item = (impl Into<String>, TensorWriteEntry)>,
        partitions: impl IntoIterator<Item = impl Into<Vec<u8>>>,
    ) -> SaveResult<Self> {
        let mut index_map = HashMap::new();
        for (name, entry) in index {
            let name = name.into();
            if name.is_empty() {
                return Err(SaveError::InvalidInput(
                    "tensor name cannot be empty".to_owned(),
                ));
            }
            index_map.insert(name, entry);
        }

        if index_map.is_empty() {
            return Err(SaveError::InvalidInput(
                "cannot create checkpoint with empty index".to_owned(),
            ));
        }

        let partitions: Vec<Vec<u8>> = partitions.into_iter().map(Into::into).collect();

        // Validate all partition references exist
        let max_partition_id = index_map
            .values()
            .map(|e| e.partition_id.as_usize())
            .max()
            .unwrap_or(0);

        if max_partition_id >= partitions.len() {
            return Err(SaveError::InvalidInput(format!(
                "tensor references partition {} but only {} partitions provided",
                max_partition_id,
                partitions.len()
            )));
        }

        Ok(Self {
            index: index_map,
            partitions,
        })
    }

    /// Returns the number of tensors in this checkpoint.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns true if there are no tensors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Returns the number of partitions.
    #[inline]
    #[must_use]
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Returns an iterator over tensor entries.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorWriteEntry)> {
        self.index.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns the tensor entry for a given name.
    #[inline]
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorWriteEntry> {
        self.index.get(name)
    }

    /// Returns true if the checkpoint contains a tensor with the given name.
    #[inline]
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Encode the index as JSON bytes.
    fn encode_index(&self) -> SaveResult<Vec<u8>> {
        if self.index.is_empty() {
            return Err(SaveError::InvalidInput(
                "Cannot write empty tensor index".to_owned(),
            ));
        }
        let mut map = serde_json::Map::with_capacity(self.index.len());
        for (name, entry) in &self.index {
            let value = serde_json::json!([
                entry.offset,
                entry.size,
                entry.shape,
                entry.stride,
                entry.dtype.as_str(),
                entry.partition_id.as_usize()
            ]);
            map.insert(name.clone(), value);
        }
        serde_json::to_vec_pretty(&map).map_err(SaveError::from)
    }

    /// Path for a partition file within a directory.
    fn partition_path(directory: &Path, partition_id: PartitionId) -> std::path::PathBuf {
        directory.join(partition_id.data_file_stem())
    }

    /// Encode a tensor index as JSON bytes (shared helper for checkpoint and converter).
    ///
    /// This is the canonical JSON serialization for the ServerlessLLM format.
    pub(crate) fn encode_index_bytes(
        index: &HashMap<String, TensorWriteEntry>,
    ) -> SaveResult<Vec<u8>> {
        if index.is_empty() {
            return Err(SaveError::InvalidInput(
                "Cannot write empty tensor index".to_owned(),
            ));
        }
        let mut map = serde_json::Map::with_capacity(index.len());
        for (name, entry) in index {
            let value = serde_json::json!([
                entry.offset,
                entry.size,
                entry.shape,
                entry.stride,
                entry.dtype.as_str(),
                entry.partition_id.as_usize()
            ]);
            map.insert(name.clone(), value);
        }
        serde_json::to_vec_pretty(&map).map_err(SaveError::from)
    }

    /// Load index from file synchronously.
    fn load_index_sync(path: &Path) -> LoadResult<Index> {
        let data = std::fs::read(path)?;
        Index::from_bytes(&data)
    }

    /// Load index from file asynchronously.
    async fn load_index_async(path: &Path) -> LoadResult<Index> {
        let engine = Tokio::new();
        let data = engine.read_file(path).await?;
        Index::from_bytes(&data)
    }
}

impl CheckpointTrait for Checkpoint {
    type Model = Model;

    fn load(path: impl AsRef<Path>, backend: Backend) -> LoadResult<Self::Model> {
        let dir = path.as_ref();
        let index = Self::load_index_sync(&dir.join("tensor_index.json"))?;
        let mut partitions = HashMap::with_capacity(index.partition_ids().len());

        match backend {
            Backend::Sync => {
                let engine = Sync::new();
                for id in index.partition_ids() {
                    let path = dir.join(id.data_file_stem());
                    let bytes = engine.read_file(&path).map_err(LoadError::from)?;
                    partitions.insert(*id, bytes.into_shared());
                }
            }
            #[cfg(target_os = "linux")]
            Backend::IoUring => {
                let engine = crate::io::io_uring::IoUring::new();
                for id in index.partition_ids() {
                    let path = dir.join(id.data_file_stem());
                    let bytes = engine.read_file(&path).map_err(LoadError::from)?;
                    partitions.insert(*id, bytes.into_shared());
                }
            }
        }

        Ok(Model::from_eager(index, partitions))
    }

    async fn aload(path: impl AsRef<Path> + Send, backend: AsyncBackend) -> LoadResult<Self::Model> {
        let dir = path.as_ref();
        let index = Self::load_index_async(&dir.join("tensor_index.json")).await?;
        let mut partitions = HashMap::with_capacity(index.partition_ids().len());

        match backend {
            AsyncBackend::Tokio => {
                let engine = Tokio::new();
                for id in index.partition_ids() {
                    let path = dir.join(id.data_file_stem());
                    let bytes = engine.read_file(&path).await.map_err(LoadError::from)?;
                    partitions.insert(*id, bytes.into_shared());
                }
            }
        }

        Ok(Model::from_eager(index, partitions))
    }

    fn open(path: impl AsRef<Path>) -> LoadResult<Self::Model> {
        let dir = path.as_ref();
        let index = Self::load_index_sync(&dir.join("tensor_index.json"))?;
        let mapper = Mmap::new();
        let mut partitions = HashMap::with_capacity(index.partition_ids().len());

        for id in index.partition_ids() {
            let path = dir.join(id.data_file_stem());
            let mmap = mapper.map_file(&path).map_err(LoadError::from)?;
            partitions.insert(*id, mmap);
        }

        Ok(Model::from_mmap(index, partitions))
    }

    fn save(&self, directory: impl AsRef<Path>) -> SaveResult<()> {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory)?;
        let engine = Sync::new();

        let index_path = directory.join("tensor_index.json");
        let index_bytes = self.encode_index()?;
        engine
            .write_file(&index_path, &index_bytes)
            .map_err(SaveError::from)?;
        engine.sync_all(&index_path).map_err(SaveError::from)?;

        for (partition_id, data) in self.partitions.iter().enumerate() {
            let path = Self::partition_path(directory, PartitionId::new(partition_id));
            engine.write_file(&path, data).map_err(SaveError::from)?;
            engine.sync_all(&path).map_err(SaveError::from)?;
        }

        Ok(())
    }

    async fn asave(&self, directory: impl AsRef<Path> + Send) -> SaveResult<()> {
        let directory = directory.as_ref();
        tokio::fs::create_dir_all(directory).await?;
        let engine = Tokio::new();

        let index_path = directory.join("tensor_index.json");
        let index_bytes = self.encode_index()?;
        engine
            .write_file(&index_path, &index_bytes)
            .await
            .map_err(SaveError::from)?;
        engine
            .sync_all(&index_path)
            .await
            .map_err(SaveError::from)?;

        for (partition_id, data) in self.partitions.iter().enumerate() {
            let path = Self::partition_path(directory, PartitionId::new(partition_id));
            engine
                .write_file(&path, data)
                .await
                .map_err(SaveError::from)?;
            engine
                .sync_all(&path)
                .await
                .map_err(SaveError::from)?;
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::traits::Model as _;
    use tempfile::TempDir;

    #[test]
    fn checkpoint_validates_empty_index() {
        let empty_index: [(String, TensorWriteEntry); 0] = [];
        let result: Result<Checkpoint, _> = Checkpoint::new(
            empty_index,
            [vec![1, 2, 3, 4]],
        );
        assert!(result.is_err());
    }

    #[test]
    fn checkpoint_validates_empty_tensor_name() {
        let entry = TensorWriteEntry::new(
            0, 4, vec![2, 2], vec![2, 1], Dtype::F32, PartitionId::new(0)
        ).unwrap();
        let result = Checkpoint::new(
            [("".to_owned(), entry)],
            [vec![1, 2, 3, 4]],
        );
        assert!(result.is_err());
    }

    #[test]
    fn checkpoint_validates_partition_references() {
        let entry = TensorWriteEntry::new(
            0, 4, vec![2, 2], vec![2, 1], Dtype::F32, PartitionId::new(5)
        ).unwrap();
        let result = Checkpoint::new(
            [("valid".to_owned(), entry)],
            [vec![1, 2, 3, 4]], // only partition 0
        );
        assert!(result.is_err());
    }

    #[test]
    fn checkpoint_validates_unknown_dtype() {
        let result = TensorWriteEntry::new(
            0, 4, vec![2, 2], vec![2, 1], Dtype::Unknown, PartitionId::new(0)
        );
        assert!(result.is_err());
    }

    #[test]
    fn checkpoint_validates_shape_stride_mismatch() {
        let result = TensorWriteEntry::new(
            0, 4, vec![2, 2], vec![2], Dtype::F32, PartitionId::new(0)
        );
        assert!(result.is_err());
    }

    #[test]
    fn checkpoint_validates_overflow() {
        let result = TensorWriteEntry::new(
            u64::MAX, 1, vec![2, 2], vec![2, 1], Dtype::F32, PartitionId::new(0)
        );
        assert!(result.is_err());
    }

    #[test]
    fn checkpoint_accessors_work() {
        let entry = TensorWriteEntry::new(
            8, 16, vec![4], vec![1], Dtype::F64, PartitionId::new(1)
        ).unwrap();
        assert_eq!(entry.offset(), 8);
        assert_eq!(entry.size(), 16);
        assert_eq!(entry.shape(), &[4]);
        assert_eq!(entry.stride(), &[1]);
        assert_eq!(entry.dtype(), Dtype::F64);
        assert_eq!(entry.partition_id(), PartitionId::new(1));
    }

    #[test]
    fn checkpoint_save_and_load_roundtrip() {
        let dir = TempDir::new().unwrap();

        let entry = TensorWriteEntry::new(
            0, 4, vec![2, 2], vec![2, 1], Dtype::F32, PartitionId::new(0)
        ).unwrap();
        let checkpoint = Checkpoint::new(
            [("test".to_owned(), entry)],
            [vec![1, 2, 3, 4]],
        ).unwrap();

        checkpoint.save(dir.path()).unwrap();

        let model = Checkpoint::load(dir.path(), Backend::Sync).unwrap();
        assert_eq!(model.len(), 1);
        assert!(model.contains("test"));

        let tensor = model.tensor("test").unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn checkpoint_iter_provides_entries() {
        let entry1 = TensorWriteEntry::new(
            0, 4, vec![2, 2], vec![2, 1], Dtype::F32, PartitionId::new(0)
        ).unwrap();
        let entry2 = TensorWriteEntry::new(
            4, 8, vec![2, 4], vec![4, 1], Dtype::F64, PartitionId::new(0)
        ).unwrap();
        let checkpoint = Checkpoint::new(
            [
                ("a".to_owned(), entry1),
                ("b".to_owned(), entry2),
            ],
            [vec![0; 12]],
        ).unwrap();

        let mut pairs: Vec<(&str, Dtype)> = checkpoint
            .iter()
            .map(|(name, entry)| (name, entry.dtype()))
            .collect();
        pairs.sort_by_key(|(name, _)| *name);

        assert_eq!(pairs, vec![("a", Dtype::F32), ("b", Dtype::F64)]);
    }
}
