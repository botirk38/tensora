//! ServerlessLLM format model.
//!
//! The model type only provides read-only access to loaded tensors. Loading is
//! owned by [`Checkpoint`](crate::formats::serverlessllm::Checkpoint).

use crate::formats::traits::Model as ModelTrait;
use crate::io::buffer::MmapRegion;
use std::collections::HashMap;
use std::sync::Arc;

use super::ids::PartitionId;
use super::index::Index;
use super::tensor::Tensor;

// ============================================================================
// Storage
// ============================================================================

#[derive(Debug, Clone)]
enum ModelStorage {
    Eager {
        index: Index,
        partitions: HashMap<PartitionId, Arc<[u8]>>,
    },
    Mmap {
        index: Index,
        partitions: HashMap<PartitionId, MmapRegion>,
    },
}

// ============================================================================
// Model
// ============================================================================

/// ServerlessLLM model with either eager or lazy (mmap) storage.
#[derive(Debug, Clone)]
pub struct Model {
    storage: ModelStorage,
}

impl Model {
    /// Build a model from an eager index and owned partition buffers.
    #[must_use]
    pub(crate) fn from_eager(index: Index, partitions: HashMap<PartitionId, Arc<[u8]>>) -> Self {
        Self {
            storage: ModelStorage::Eager { index, partitions },
        }
    }

    /// Build a model from an index and memory-mapped partition regions.
    #[must_use]
    pub(crate) fn from_mmap(
        index: Index,
        partitions: HashMap<PartitionId, MmapRegion>,
    ) -> Self {
        Self {
            storage: ModelStorage::Mmap { index, partitions },
        }
    }

    /// Returns true if this model uses lazy (mmap-backed) storage.
    #[inline]
    #[must_use]
    pub fn is_lazy(&self) -> bool {
        matches!(self.storage, ModelStorage::Mmap { .. })
    }

    /// Returns a view of the tensor with the given name, or `None` if missing.
    #[inline]
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<Tensor<'_>> {
        match &self.storage {
            ModelStorage::Eager { index, partitions } => {
                let desc = index.get(name)?;
                let partition = partitions.get(&desc.partition_id())?;
                let start = usize::try_from(desc.offset()).ok()?;
                let end = start.checked_add(desc.size())?;
                let data = partition.get(start..end)?;
                Some(Tensor::eager(desc, data))
            }
            ModelStorage::Mmap { index, partitions } => {
                let desc = index.get(name)?;
                let partition = partitions.get(&desc.partition_id())?;
                let start = usize::try_from(desc.offset()).ok()?;
                let end = start.checked_add(desc.size())?;
                if end > partition.len() {
                    return None;
                }
                let region = partition.subregion(start, desc.size())?;
                Some(Tensor::mmap(desc, region))
            }
        }
    }

    /// Returns the number of tensors.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.storage {
            ModelStorage::Eager { index, .. } => index.len(),
            ModelStorage::Mmap { index, .. } => index.len(),
        }
    }

    /// Returns true if there are no tensors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over tensor names.
    #[allow(dead_code)]
    fn tensor_names_iter(&self) -> Box<dyn ExactSizeIterator<Item = &str> + '_> {
        match &self.storage {
            ModelStorage::Eager { index, .. } => Box::new(index.tensor_names_iter()),
            ModelStorage::Mmap { index, .. } => Box::new(index.tensor_names_iter()),
        }
    }
}

impl ModelTrait for Model {
    type Tensor<'a>
        = Tensor<'a>
    where
        Self: 'a;

    type Names<'a>
        = std::iter::Map<std::slice::Iter<'a, Arc<str>>, fn(&'a Arc<str>) -> &'a str>
    where
        Self: 'a;

    #[inline]
    fn len(&self) -> usize {
        Model::len(self)
    }

    #[inline]
    fn tensor_names(&self) -> Self::Names<'_> {
        match &self.storage {
            ModelStorage::Eager { index, .. } => index.tensor_names(),
            ModelStorage::Mmap { index, .. } => index.tensor_names(),
        }
    }

    #[inline]
    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
        Model::tensor(self, name)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::tensor::Dtype;
    use crate::formats::serverlessllm::checkpoint::{Checkpoint, TensorWriteEntry};
    use crate::formats::serverlessllm::ids::PartitionId;
    use crate::formats::traits::Checkpoint as _;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn sample_checkpoint() -> Checkpoint {
        let mut index = HashMap::new();
        index.insert(
            "w".to_owned(),
            TensorWriteEntry::new(
                0,
                4,
                vec![2, 2],
                vec![2, 1],
                Dtype::F32,
                PartitionId::new(0),
            ).unwrap(),
        );
        Checkpoint::new(index, vec![vec![1, 2, 3, 4]]).unwrap()
    }

    #[test]
    fn model_empty() {
        let model = Model {
            storage: ModelStorage::Eager {
                index: Index::new(),
                partitions: HashMap::new(),
            },
        };
        assert!(model.is_empty());
        assert!(!model.is_lazy());
    }

    #[test]
    fn model_load_sync() {
        let dir = TempDir::new().unwrap();
        let checkpoint = sample_checkpoint();
        checkpoint.save(dir.path()).unwrap();

        let model = Checkpoint::load(dir.path(), crate::formats::Backend::Sync).unwrap();
        assert_eq!(model.len(), 1);
        assert!(model.contains("w"));

        let tensor = model.tensor("w").unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), Dtype::F32);
        assert_eq!(tensor.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn model_open_mmap() {
        let dir = TempDir::new().unwrap();
        let checkpoint = sample_checkpoint();
        checkpoint.save(dir.path()).unwrap();

        let model = Checkpoint::open(dir.path()).unwrap();
        assert!(model.is_lazy());
        assert_eq!(model.len(), 1);

        let tensor = model.tensor("w").unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), Dtype::F32);
        assert_eq!(tensor.data(), &[1, 2, 3, 4]);
    }
}
