//! ServerlessLLM format model.
//!
//! The model type only provides read-only access to loaded tensors. Loading is
//! owned by [`Checkpoint`](crate::formats::serverlessllm::Checkpoint).

use fastio::MmapRegion;
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
    pub(crate) fn from_eager(index: Index, partitions: HashMap<PartitionId, Arc<[u8]>>) -> Self {
        Self {
            storage: ModelStorage::Eager { index, partitions },
        }
    }

    pub(crate) fn from_mmap(index: Index, partitions: HashMap<PartitionId, MmapRegion>) -> Self {
        Self {
            storage: ModelStorage::Mmap { index, partitions },
        }
    }

    /// Returns true if this model uses lazy (mmap-backed) storage.
    #[must_use]
    pub fn is_lazy(&self) -> bool {
        matches!(self.storage, ModelStorage::Mmap { .. })
    }

    /// Returns true if the model contains the named tensor.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        match &self.storage {
            ModelStorage::Eager { index, .. } => index.contains(name),
            ModelStorage::Mmap { index, .. } => index.contains(name),
        }
    }
}

impl crate::formats::traits::Model for Model {
    type Tensor<'a>
        = Tensor<'a>
    where
        Self: 'a;
    type Names<'a>
        = std::iter::Map<std::slice::Iter<'a, Arc<str>>, fn(&'a Arc<str>) -> &'a str>
    where
        Self: 'a;

    fn len(&self) -> usize {
        match &self.storage {
            ModelStorage::Eager { index, .. } => index.len(),
            ModelStorage::Mmap { index, .. } => index.len(),
        }
    }

    fn tensor_names(&self) -> Self::Names<'_> {
        match &self.storage {
            ModelStorage::Eager { index, .. } => index.tensor_names(),
            ModelStorage::Mmap { index, .. } => index.tensor_names(),
        }
    }

    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
        match &self.storage {
            ModelStorage::Eager { index, partitions } => {
                let pt = index.get(name)?;
                let partition = partitions.get(&pt.partition_id())?;
                let start = usize::try_from(pt.offset()).ok()?;
                let end = start.checked_add(usize::try_from(pt.size()).ok()?)?;
                Some(Tensor::eager(pt, partition.get(start..end)?))
            }
            ModelStorage::Mmap { index, partitions } => {
                let pt = index.get(name)?;
                let partition = partitions.get(&pt.partition_id())?;
                let start = usize::try_from(pt.offset()).ok()?;
                let size = usize::try_from(pt.size()).ok()?;
                let end = start.checked_add(size)?;
                if end > partition.len() {
                    return None;
                }
                Some(Tensor::mmap(pt, partition.subregion(start, size)?))
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{Index, Model as ServerlessLLMModel, ModelStorage};
    use crate::formats::serverlessllm::checkpoint::Checkpoint as ServerlessLLMCheckpoint;
    use crate::formats::serverlessllm::ids::PartitionId;
    use crate::formats::serverlessllm::tensor::TensorEntry;
    use crate::formats::tensor::{Dtype, TensorMeta};
    use crate::formats::traits::{Checkpoint, Model, Tensor as TensorTrait};
    use tempfile::TempDir;

    fn sample_checkpoint() -> ServerlessLLMCheckpoint {
        let meta = TensorMeta::new(0, 4, vec![2usize, 2], vec![2usize, 1], Dtype::F32).unwrap();
        let pt = TensorEntry::new(meta, PartitionId::new(0));
        ServerlessLLMCheckpoint::new([("w".to_owned(), pt)], [vec![1u8, 2, 3, 4]]).unwrap()
    }

    #[test]
    fn model_empty() {
        let model = ServerlessLLMModel {
            storage: ModelStorage::Eager {
                index: Index::new(),
                partitions: HashMap::new(),
            },
        };
        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
        assert!(!model.is_lazy());
    }

    #[test]
    fn model_load_sync() {
        let dir = TempDir::new().unwrap();
        sample_checkpoint().save(dir.path()).unwrap();

        let model =
            ServerlessLLMCheckpoint::load(dir.path(), crate::formats::Backend::Sync).unwrap();
        assert_eq!(model.len(), 1);
        assert!(model.contains("w"));

        let tensor = model.tensor("w").unwrap();
        assert_eq!(TensorTrait::shape(&tensor), &[2, 2]);
        assert_eq!(TensorTrait::dtype(&tensor), Dtype::F32);
        assert_eq!(TensorTrait::data(&tensor), &[1, 2, 3, 4]);
    }

    #[test]
    fn model_open_mmap() {
        let dir = TempDir::new().unwrap();
        sample_checkpoint().save(dir.path()).unwrap();

        let model = ServerlessLLMCheckpoint::open(dir.path()).unwrap();
        assert!(model.is_lazy());
        assert_eq!(model.len(), 1);

        let tensor = model.tensor("w").unwrap();
        assert_eq!(TensorTrait::shape(&tensor), &[2, 2]);
        assert_eq!(TensorTrait::dtype(&tensor), Dtype::F32);
        assert_eq!(TensorTrait::data(&tensor), &[1, 2, 3, 4]);
    }
}
