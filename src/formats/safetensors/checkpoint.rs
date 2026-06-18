//! `SafeTensors` format checkpoint.
//!
//! This module provides a format-specific checkpoint type that loads and saves
//! SafeTensors models. Loading produces a [`Model`](crate::formats::safetensors::Model);
//! saving writes a single `.safetensors` file.
//!
//! # Example
//!
//! ```rust,ignore
//! use tensora::formats::safetensors::{Checkpoint, Dtype, TensorWriteData};
//!
//! let data = vec![0u8; 4];
//! let checkpoint = Checkpoint::new(
//!     vec![TensorWriteData::new(
//!         "weight",
//!         data,
//!         vec![1, 1],
//!         Dtype::F32,
//!     ).unwrap()],
//!     None,
//! );
//!
//! checkpoint.save("model.safetensors").unwrap();
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

use std::fs;
use std::path::{Path, PathBuf};

use super::model::{Model, ShardData};
pub use safetensors::tensor::View;

/// Convenience alias for custom metadata passed to `SafeTensors`.
pub type MetadataMap = HashMap<String, String>;

// ============================================================================
// TensorWriteData
// ============================================================================

/// Input data for writing a SafeTensors checkpoint.
#[derive(Debug, Clone)]
pub struct TensorWriteData {
    name: String,
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: Dtype,
}

impl TensorWriteData {
    /// Create a new validated tensor write entry.
    ///
    /// # Errors
    ///
    /// Returns `SaveError::InvalidInput` if:
    /// - `name` is empty
    /// - `dtype` is `Dtype::Unknown`
    pub fn new(
        name: impl Into<String>,
        data: impl Into<Vec<u8>>,
        shape: impl Into<Vec<usize>>,
        dtype: Dtype,
    ) -> SaveResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(SaveError::InvalidInput(
                "tensor name cannot be empty".to_owned(),
            ));
        }

        if dtype == Dtype::Unknown {
            return Err(SaveError::InvalidInput(
                "cannot use Unknown dtype for tensor".to_owned(),
            ));
        }

        let data = data.into();
        let shape = shape.into();

        Ok(Self {
            name,
            data,
            shape,
            dtype,
        })
    }

    /// Returns the tensor name.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tensor data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns the tensor shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the dtype.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
}

// ============================================================================
// Checkpoint
// ============================================================================

/// A SafeTensors checkpoint, ready to serialize.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    tensors: Vec<TensorWriteData>,
    metadata: Option<MetadataMap>,
}

impl Checkpoint {
    /// Create a new checkpoint from tensor data and optional metadata.
    ///
    /// # Errors
    ///
    /// Returns `SaveError::InvalidInput` if the tensor list is empty.
    pub fn new(
        tensors: impl IntoIterator<Item = TensorWriteData>,
        metadata: Option<MetadataMap>,
    ) -> SaveResult<Self> {
        let tensors: Vec<TensorWriteData> = tensors.into_iter().collect();
        if tensors.is_empty() {
            return Err(SaveError::InvalidInput(
                "cannot create checkpoint with empty tensor list".to_owned(),
            ));
        }

        Ok(Self {
            tensors,
            metadata,
        })
    }

    /// Returns the number of tensors.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true if there are no tensors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns an iterator over tensor write data.
    pub fn iter(&self) -> impl Iterator<Item = &TensorWriteData> {
        self.tensors.iter()
    }

    /// Serialize the checkpoint into an owned byte buffer.
    pub fn to_bytes(&self) -> SaveResult<Vec<u8>> {
        let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = self.views()?;
        safetensors::serialize(views, self.metadata.clone()).map_err(SaveError::from)
    }

    /// Build `safetensors` tensor views from the checkpoint data.
    fn views(&self) -> SaveResult<Vec<(&str, safetensors::tensor::TensorView<'_>)>> {
        self.tensors
            .iter()
            .map(|t| {
                let dtype = safetensors::Dtype::from(t.dtype);
                let view = safetensors::tensor::TensorView::new(dtype, t.shape.clone(), &t.data)
                    .map_err(|e| SaveError::InvalidInput(e.to_string()))?;
                Ok((t.name.as_str(), view))
            })
            .collect()
    }

    /// Discover `.safetensors` shard files in a directory.
    fn discover_shards(path: impl AsRef<Path>) -> LoadResult<Vec<PathBuf>> {
        let path = path.as_ref();
        if path.is_file() {
            return Err(LoadError::InvalidMetadata(format!(
                "SafeTensors loads a directory of .safetensors shards, got file path {}",
                path.display()
            )));
        }
        if !path.is_dir() {
            return Err(LoadError::InvalidMetadata(format!(
                "SafeTensors loads a directory of .safetensors shards, got path {}",
                path.display()
            )));
        }

        let mut shard_paths = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if !entry_path.is_file() {
                continue;
            }

            let Some(name) = entry_path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.ends_with(".safetensors") {
                shard_paths.push(entry_path);
            }
        }

        shard_paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        if shard_paths.is_empty() {
            return Err(LoadError::InvalidMetadata(format!(
                "no .safetensors files found in {}",
                path.display()
            )));
        }

        Ok(shard_paths)
    }
}

impl CheckpointTrait for Checkpoint {
    type Model = Model;

    fn load(path: impl AsRef<Path>, backend: Backend) -> LoadResult<Self::Model> {
        let paths = Self::discover_shards(path)?;
        let mut shards = Vec::with_capacity(paths.len());

        match backend {
            Backend::Sync => {
                let engine = Sync::new();
                for path in paths {
                    let bytes = engine.read_file(&path).map_err(LoadError::from)?;
                    shards.push(ShardData::parse(bytes)?);
                }
            }
            #[cfg(target_os = "linux")]
            Backend::IoUring => {
                let engine = crate::io::io_uring::IoUring::new();
                for path in paths {
                    let bytes = engine.read_file(&path).map_err(LoadError::from)?;
                    shards.push(ShardData::parse(bytes)?);
                }
            }
        }

        Model::from_shards(shards)
    }

    async fn aload(path: impl AsRef<Path> + Send, backend: AsyncBackend) -> LoadResult<Self::Model> {
        let paths = Self::discover_shards(path)?;
        let mut shards = Vec::with_capacity(paths.len());

        match backend {
            AsyncBackend::Tokio => {
                let engine = Tokio::new();
                for path in paths {
                    let bytes = engine.read_file(&path).await.map_err(LoadError::from)?;
                    shards.push(ShardData::parse(bytes)?);
                }
            }
        }

        Model::from_shards(shards)
    }

    fn open(path: impl AsRef<Path>) -> LoadResult<Self::Model> {
        let paths = Self::discover_shards(path)?;
        let mapper = Mmap::new();
        let mut shards = Vec::with_capacity(paths.len());

        for path in paths {
            let mmap = mapper.map_file(&path).map_err(LoadError::from)?;
            shards.push(ShardData::parse(mmap)?);
        }

        Model::from_shards(shards)
    }

    fn save(&self, path: impl AsRef<Path>) -> SaveResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = self.views()?;
        safetensors::serialize_to_file(views, self.metadata.clone(), path)
            .map_err(SaveError::from)
    }

    async fn asave(&self, path: impl AsRef<Path> + Send) -> SaveResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            tokio::fs::create_dir_all(parent).await?;
        }
        let bytes = self.to_bytes()?;
        let engine = Tokio::new();
        engine
            .write_file(path, &bytes)
            .await
            .map_err(SaveError::from)?;
        engine.sync_all(path).await.map_err(SaveError::from)
    }
}

impl From<Dtype> for safetensors::Dtype {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::Bool => Self::BOOL,
            Dtype::U8 => Self::U8,
            Dtype::I8 => Self::I8,
            Dtype::I16 => Self::I16,
            Dtype::U16 => Self::U16,
            Dtype::F16 => Self::F16,
            Dtype::F32 => Self::F32,
            Dtype::F64 => Self::F64,
            Dtype::I32 => Self::I32,
            Dtype::I64 => Self::I64,
            Dtype::U32 => Self::U32,
            Dtype::U64 => Self::U64,
            Dtype::Bf16 => Self::BF16,
            Dtype::Unknown => Self::F32,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::traits::{Checkpoint as _, Model as _, Tensor as _};

    fn sample_checkpoint() -> Checkpoint {
        Checkpoint::new(
            vec![
                TensorWriteData::new("a", vec![0u8; 4], vec![1], Dtype::F32).unwrap(),
                TensorWriteData::new("b", vec![0u8; 8], vec![2], Dtype::F64).unwrap(),
            ],
            None,
        )
        .unwrap()
    }

    #[test]
    fn checkpoint_validates_empty_tensor_list() {
        let result: Result<Checkpoint, _> = Checkpoint::new(vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn tensor_write_data_validates_empty_name() {
        let result = TensorWriteData::new("", vec![0u8; 4], vec![1], Dtype::F32);
        assert!(result.is_err());
    }

    #[test]
    fn tensor_write_data_validates_unknown_dtype() {
        let result = TensorWriteData::new("test", vec![0u8; 4], vec![1], Dtype::Unknown);
        assert!(result.is_err());
    }

    #[test]
    fn tensor_write_data_accessors_work() {
        let data = TensorWriteData::new("weight", vec![1, 2, 3, 4], vec![2, 2], Dtype::F32).unwrap();
        assert_eq!(data.name(), "weight");
        assert_eq!(data.data(), &[1, 2, 3, 4]);
        assert_eq!(data.shape(), &[2, 2]);
        assert_eq!(data.dtype(), Dtype::F32);
    }

    #[test]
    fn checkpoint_iter_provides_entries() {
        let cp = sample_checkpoint();
        let names: Vec<&str> = cp.iter().map(|t| t.name()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn checkpoint_len_and_is_empty() {
        let cp = sample_checkpoint();
        assert_eq!(cp.len(), 2);
        assert!(!cp.is_empty());
    }
}
