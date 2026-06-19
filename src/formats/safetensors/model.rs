//! `SafeTensors` format model.
//!
//! The model type only provides read-only access to loaded tensors. Loading is
//! owned by [`Checkpoint`](crate::formats::safetensors::Checkpoint).

use crate::formats::error::{LoadError, LoadResult};
use crate::formats::tensor::Dtype;
use crate::formats::traits::Model as ModelTrait;
use crate::io::buffer::{MmapRegion, OwnedBytes};
use safetensors::tensor::Metadata;
use std::collections::HashMap;
use std::sync::Arc;

use super::ids::ShardId;
use super::tensor::Tensor;

// ============================================================================
// Storage
// ============================================================================

#[derive(Debug, Clone)]
pub(crate) enum Backing {
    Owned(Arc<[u8]>),
    Mmap(MmapRegion),
}

impl Backing {
    #[inline]
    #[must_use]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Mmap(region) => region.as_slice(),
        }
    }
}

impl From<OwnedBytes> for Backing {
    #[inline]
    fn from(buffer: OwnedBytes) -> Self {
        Self::Owned(buffer.into_shared())
    }
}

impl From<MmapRegion> for Backing {
    #[inline]
    fn from(region: MmapRegion) -> Self {
        Self::Mmap(region)
    }
}

impl From<Vec<u8>> for Backing {
    #[inline]
    fn from(bytes: Vec<u8>) -> Self {
        Self::Owned(bytes.into())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ShardData {
    pub(crate) metadata: Metadata,
    pub(crate) data_start: usize,
    pub(crate) backing: Backing,
}

impl ShardData {
    /// Parse a SafeTensors shard from raw bytes or an mmap region.
    pub(crate) fn parse(backing: impl Into<Backing>) -> LoadResult<Self> {
        const N_LEN: usize = std::mem::size_of::<u64>();
        let backing = backing.into();
        let (header_json_len, metadata) = safetensors::SafeTensors::read_metadata(backing.as_slice())?;
        Ok(Self {
            metadata,
            data_start: header_json_len + N_LEN,
            backing,
        })
    }
}

#[derive(Debug, Clone)]
enum ModelStorage {
    Single {
        shard: ShardData,
        tensor_names: Arc<[Arc<str>]>,
    },
    Sharded {
        shards: Vec<ShardData>,
        tensor_shards: HashMap<Arc<str>, ShardId>,
        tensor_names: Arc<[Arc<str>]>,
    },
}

// ============================================================================
// Model
// ============================================================================

/// SafeTensors model with either eager or lazy (mmap) storage.
#[derive(Debug, Clone)]
pub struct Model {
    storage: ModelStorage,
}

impl Model {
    /// Build a model from parsed shards.
    ///
    /// Used by [`Checkpoint`](crate::formats::safetensors::Checkpoint) after it
    /// owns the loading process.
    pub(crate) fn from_shards(shards: Vec<ShardData>) -> LoadResult<Self> {
        if shards.len() == 1 {
            let shard = shards.into_iter().next().unwrap();
            let mut tensor_names: Vec<Arc<str>> = shard
                .metadata
                .offset_keys()
                .into_iter()
                .map(Arc::from)
                .collect();
            tensor_names.sort_unstable();

            return Ok(Self {
                storage: ModelStorage::Single {
                    shard,
                    tensor_names: tensor_names.into(),
                },
            });
        }

        let mut tensor_shards: HashMap<Arc<str>, ShardId> = HashMap::new();
        let mut tensor_names: Vec<Arc<str>> = Vec::new();

        for (i, shard) in shards.iter().enumerate() {
            let shard_id = ShardId::new(i);
            for name in shard.metadata.offset_keys() {
                let name: Arc<str> = Arc::from(name.as_str());
                if tensor_shards.insert(name.clone(), shard_id).is_some() {
                    return Err(LoadError::InvalidMetadata(format!(
                        "duplicate tensor name across shards: {name}"
                    )));
                }
                tensor_names.push(name);
            }
        }

        tensor_names.sort_unstable();

        Ok(Self {
            storage: ModelStorage::Sharded {
                shards,
                tensor_shards,
                tensor_names: tensor_names.into(),
            },
        })
    }

    /// Returns true if this model uses lazy (mmap-backed) storage.
    #[must_use]
    pub fn is_lazy(&self) -> bool {
        match &self.storage {
            ModelStorage::Single { shard, .. } => matches!(shard.backing, Backing::Mmap(_)),
            ModelStorage::Sharded { shards, .. } => shards.iter().any(|s| matches!(s.backing, Backing::Mmap(_))),
        }
    }
}

fn arc_to_str(arc: &Arc<str>) -> &str { arc.as_ref() }

impl ModelTrait for Model {
    type Tensor<'a> = Tensor<'a> where Self: 'a;
    type Names<'a> = std::iter::Map<std::slice::Iter<'a, Arc<str>>, fn(&'a Arc<str>) -> &'a str>
        where Self: 'a;

    fn len(&self) -> usize {
        match &self.storage {
            ModelStorage::Single { tensor_names, .. } => tensor_names.len(),
            ModelStorage::Sharded { tensor_names, .. } => tensor_names.len(),
        }
    }

    fn tensor_names(&self) -> Self::Names<'_> {
        match &self.storage {
            ModelStorage::Single { tensor_names, .. } => tensor_names.iter().map(arc_to_str),
            ModelStorage::Sharded { tensor_names, .. } => tensor_names.iter().map(arc_to_str),
        }
    }

    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
        let (metadata, data_start, backing) = match &self.storage {
            ModelStorage::Single { shard, .. } => (&shard.metadata, shard.data_start, &shard.backing),
            ModelStorage::Sharded { shards, tensor_shards, .. } => {
                let shard = &shards[tensor_shards.get(name)?.as_usize()];
                (&shard.metadata, shard.data_start, &shard.backing)
            }
        };
        let info = metadata.info(name)?;
        let (start, end) = info.data_offsets;
        let data = backing.as_slice()[data_start..].get(start..end)?;
        Some(Tensor::new(&info.shape, Dtype::from(info.dtype), data))
    }
}

// ============================================================================
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::traits::Tensor as _;
    use crate::io::buffer::OwnedBytes;

    fn dummy_shard_data() -> ShardData {
        // Minimal valid SafeTensors: 8-byte length (LE u64) + header JSON
        // Header length = 2 bytes for "{}"
        let header = b"{}";
        let len = header.len() as u64;
        let mut data = Vec::with_capacity(8 + header.len());
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(header);
        ShardData::parse(OwnedBytes::from_vec(data)).expect("parse minimal shard")
    }

    #[test]
    fn model_from_single_shard() {
        let shard = dummy_shard_data();
        let model = Model::from_shards(vec![shard]).expect("single shard ok");
        assert!(model.is_empty());
        assert!(model.tensor_names().next().is_none());
    }

    #[test]
    fn model_len_matches_tensor_count() {
        // Create a shard with known tensor names for testing
        // This requires a valid SafeTensors header with tensors
        // For now, just verify empty model has len 0
        let shard = dummy_shard_data();
        let model = Model::from_shards(vec![shard]).expect("parse");
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn model_is_empty_when_no_tensors() {
        let shard = dummy_shard_data();
        let model = Model::from_shards(vec![shard]).expect("parse");
        assert!(model.is_empty());
    }

    #[test]
    fn tensor_returns_none_for_missing() {
        let shard = dummy_shard_data();
        let model = Model::from_shards(vec![shard]).expect("parse");
        assert!(model.tensor("nonexistent").is_none());
    }

    #[test]
    fn model_contains_uses_tensor_lookup() {
        let shard = dummy_shard_data();
        let model = Model::from_shards(vec![shard]).expect("parse");
        assert!(!model.contains("any"));
    }

    #[test]
    fn tensor_stride_returns_none() {
        // SafeTensors tensors do not store explicit stride
        let shape = vec![2, 3, 4];
        let data = vec![0u8; 2 * 3 * 4 * 4]; // f32
        let tensor = Tensor::new(&shape, Dtype::F32, &data);
        assert_eq!(tensor.stride(), None);
    }
}
