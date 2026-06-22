//! `ServerlessLLM` format checkpoint.
//!
//! Provides [`Checkpoint`], which loads and saves ServerlessLLM models.
//! Loading produces a [`Model`]; saving writes `tensor_index.json` plus the
//! partition data files (`tensor.data_0`, `tensor.data_1`, …).
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
use crate::formats::{AsyncBackend, Backend};
use fastio::mmap::Mmap;
use fastio::sync::SyncIo;
use fastio::tokio::Tokio;
use fastio::{AsyncIo, BlockingIo, MmapIo};
use std::collections::HashMap;
use std::path::Path;

use super::ids::PartitionId;
use super::index::Index;
use super::model::Model;
use super::tensor::TensorEntry;

// ============================================================================
// Checkpoint
// ============================================================================

/// A ServerlessLLM checkpoint, ready to serialize.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    index: HashMap<String, TensorEntry>,
    partitions: Vec<Vec<u8>>,
}

impl Checkpoint {
    /// Create a new validated checkpoint.
    ///
    /// # Errors
    ///
    /// Returns `SaveError::InvalidInput` if:
    /// - `index` is empty
    /// - any tensor name is empty
    /// - a tensor references a partition index that is out of bounds for `partitions`
    pub fn new(
        index: impl IntoIterator<Item = (impl Into<String>, TensorEntry)>,
        partitions: impl IntoIterator<Item = impl Into<Vec<u8>>>,
    ) -> SaveResult<Self> {
        let mut index_map = HashMap::new();
        for (name, pt) in index {
            let name = name.into();
            if name.is_empty() {
                return Err(SaveError::InvalidInput(
                    "tensor name cannot be empty".to_owned(),
                ));
            }
            index_map.insert(name, pt);
        }

        if index_map.is_empty() {
            return Err(SaveError::InvalidInput(
                "cannot create checkpoint with empty index".to_owned(),
            ));
        }

        let partitions: Vec<Vec<u8>> = partitions.into_iter().map(Into::into).collect();

        let max_partition_id = index_map
            .values()
            .map(|pt| pt.partition_id().as_usize())
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

    /// Encode the index map as canonical `tensor_index.json` bytes.
    ///
    /// Called by `save`/`asave` and the converter's materialize step.
    pub(crate) fn encode_index(index: &HashMap<String, TensorEntry>) -> SaveResult<Vec<u8>> {
        if index.is_empty() {
            return Err(SaveError::InvalidInput(
                "cannot write empty tensor index".to_owned(),
            ));
        }
        let mut map = serde_json::Map::with_capacity(index.len());
        for (name, e) in index {
            map.insert(
                name.clone(),
                serde_json::json!([
                    e.offset(),
                    e.size(),
                    e.shape(),
                    e.stride(),
                    e.dtype().as_str(),
                    e.partition_id().as_usize()
                ]),
            );
        }
        serde_json::to_vec_pretty(&map).map_err(SaveError::from)
    }
}

// ============================================================================
// Checkpoint trait impl
// ============================================================================

impl crate::formats::traits::Checkpoint for Checkpoint {
    type Model = Model;

    fn load(path: impl AsRef<Path>, backend: Backend) -> LoadResult<Self::Model> {
        let dir = path.as_ref();
        let index = Index::from_bytes(&std::fs::read(dir.join("tensor_index.json"))?)?;
        let mut partitions = HashMap::with_capacity(index.partition_ids().len());

        match backend {
            Backend::Sync => {
                let engine = SyncIo::new();
                for id in index.partition_ids() {
                    let bytes = engine
                        .read_file(&dir.join(id.data_file_stem()))
                        .map_err(LoadError::from)?;
                    partitions.insert(*id, bytes.into_shared());
                }
            }
            #[cfg(target_os = "linux")]
            Backend::IoUring => {
                let engine = fastio::io_uring::IoUring::new();
                for id in index.partition_ids() {
                    let bytes = engine
                        .read_file(&dir.join(id.data_file_stem()))
                        .map_err(LoadError::from)?;
                    partitions.insert(*id, bytes.into_shared());
                }
            }
        }

        Ok(Model::from_eager(index, partitions))
    }

    async fn aload(
        path: impl AsRef<Path> + Send,
        backend: AsyncBackend,
    ) -> LoadResult<Self::Model> {
        let dir = path.as_ref();
        let index = Index::from_bytes(
            &Tokio::new()
                .read_file(&dir.join("tensor_index.json"))
                .await?,
        )?;
        let mut partitions = HashMap::with_capacity(index.partition_ids().len());

        match backend {
            AsyncBackend::Tokio => {
                let engine = Tokio::new();
                for id in index.partition_ids() {
                    let bytes = engine
                        .read_file(&dir.join(id.data_file_stem()))
                        .await
                        .map_err(LoadError::from)?;
                    partitions.insert(*id, bytes.into_shared());
                }
            }
        }

        Ok(Model::from_eager(index, partitions))
    }

    fn open(path: impl AsRef<Path>) -> LoadResult<Self::Model> {
        let dir = path.as_ref();
        let index = Index::from_bytes(&std::fs::read(dir.join("tensor_index.json"))?)?;
        let mapper = Mmap::new();
        let mut partitions = HashMap::with_capacity(index.partition_ids().len());

        for id in index.partition_ids() {
            let mmap = mapper
                .map_file(&dir.join(id.data_file_stem()))
                .map_err(LoadError::from)?;
            partitions.insert(*id, mmap);
        }

        Ok(Model::from_mmap(index, partitions))
    }

    fn save(&self, directory: impl AsRef<Path>) -> SaveResult<()> {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory)?;
        let engine = SyncIo::new();

        let index_path = directory.join("tensor_index.json");
        engine
            .write_file(&index_path, &Self::encode_index(&self.index)?)
            .map_err(SaveError::from)?;
        engine.sync_all(&index_path).map_err(SaveError::from)?;

        for (id, data) in self.partitions.iter().enumerate() {
            let path = directory.join(PartitionId::new(id).data_file_stem());
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
        engine
            .write_file(&index_path, &Self::encode_index(&self.index)?)
            .await
            .map_err(SaveError::from)?;
        engine
            .sync_all(&index_path)
            .await
            .map_err(SaveError::from)?;

        for (id, data) in self.partitions.iter().enumerate() {
            let path = directory.join(PartitionId::new(id).data_file_stem());
            engine
                .write_file(&path, data)
                .await
                .map_err(SaveError::from)?;
            engine.sync_all(&path).await.map_err(SaveError::from)?;
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::Checkpoint as ServerlessLLMCheckpoint;
    use crate::formats::Backend;
    use crate::formats::error::SaveResult;
    use crate::formats::serverlessllm::ids::PartitionId;
    use crate::formats::serverlessllm::tensor::TensorEntry;
    use crate::formats::tensor::{Dtype, TensorMeta};
    use crate::formats::traits::{Checkpoint, Model};
    use tempfile::TempDir;

    fn entry(
        range: (u64, u64),
        shape: Vec<usize>,
        stride: Vec<usize>,
        dtype: Dtype,
        pid: usize,
    ) -> TensorEntry {
        let (offset, size) = range;
        let meta = TensorMeta::new(offset, size, shape, stride, dtype).unwrap();
        TensorEntry::new(meta, PartitionId::new(pid))
    }

    #[test]
    fn validates_empty_index() {
        let result: SaveResult<ServerlessLLMCheckpoint> =
            ServerlessLLMCheckpoint::new(std::iter::empty::<(String, TensorEntry)>(), [vec![1u8]]);
        assert!(result.is_err());
    }

    #[test]
    fn validates_empty_tensor_name() {
        let result = ServerlessLLMCheckpoint::new(
            [(
                "".to_owned(),
                entry((0, 4), vec![2, 2], vec![2, 1], Dtype::F32, 0),
            )],
            [vec![1u8, 2, 3, 4]],
        );
        assert!(result.is_err());
    }

    #[test]
    fn validates_partition_out_of_bounds() {
        let result = ServerlessLLMCheckpoint::new(
            [(
                "w".to_owned(),
                entry((0, 4), vec![2, 2], vec![2, 1], Dtype::F32, 5),
            )],
            [vec![1u8, 2, 3, 4]], // only partition 0 provided
        );
        assert!(result.is_err());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = TempDir::new().unwrap();
        let cp = ServerlessLLMCheckpoint::new(
            [(
                "test".to_owned(),
                entry((0, 4), vec![2, 2], vec![2, 1], Dtype::F32, 0),
            )],
            [vec![1u8, 2, 3, 4]],
        )
        .unwrap();

        cp.save(dir.path()).unwrap();

        let model = ServerlessLLMCheckpoint::load(dir.path(), Backend::Sync).unwrap();
        assert_eq!(model.len(), 1);
        assert!(model.contains("test"));

        let tensor = model.tensor("test").unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn multi_tensor_roundtrip() {
        let dir = TempDir::new().unwrap();
        ServerlessLLMCheckpoint::new(
            [
                (
                    "a".to_owned(),
                    entry((0, 4), vec![2, 2], vec![2, 1], Dtype::F32, 0),
                ),
                (
                    "b".to_owned(),
                    entry((4, 8), vec![2, 4], vec![4, 1], Dtype::F64, 0),
                ),
            ],
            [vec![0u8; 12]],
        )
        .unwrap()
        .save(dir.path())
        .unwrap();

        let model = ServerlessLLMCheckpoint::load(dir.path(), Backend::Sync).unwrap();
        assert_eq!(model.len(), 2);
        assert_eq!(model.tensor("a").unwrap().dtype(), Dtype::F32);
        assert_eq!(model.tensor("b").unwrap().dtype(), Dtype::F64);
    }
}
