//! `SafeTensors` format checkpoint.
//!
//! Provides loading from a directory of `.safetensors` files and saving to a
//! single `.safetensors` file.
//!
//! # Example
//!
//! ```rust,ignore
//! use tensora::formats::safetensors::{Checkpoint, TensorEntry};
//! use tensora::formats::tensor::Dtype;
//!
//! let checkpoint = Checkpoint::new(
//!     vec![TensorEntry::new("weight", vec![0u8; 4], vec![1usize, 1], Dtype::F32).unwrap()],
//!     None,
//! ).unwrap();
//!
//! checkpoint.save("model.safetensors").unwrap();
//! ```

use crate::formats::error::{LoadError, LoadResult, SaveError, SaveResult};
use crate::formats::tensor::Dtype;
use crate::formats::{AsyncBackend, Backend};
use fastio::{mmap, sync, tokio as fastio_tokio};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::model::{FileData, Model};
use super::tensor::TensorEntry;

/// Convenience alias for custom metadata passed to `SafeTensors`.
pub type MetadataMap = HashMap<String, String>;

// ============================================================================
// Checkpoint
// ============================================================================

/// A SafeTensors checkpoint, ready to serialize.
///
/// Tensors are supplied as [`TensorEntry`] values.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    tensors: Vec<TensorEntry>,
    metadata: Option<MetadataMap>,
}

impl Checkpoint {
    /// Create a new checkpoint from [`TensorEntry`] values and optional metadata.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::InvalidInput`] if the tensor list is empty.
    /// Per-tensor validation (empty name, Unknown dtype) is performed by
    /// [`TensorEntry::new`].
    pub fn new(
        tensors: impl IntoIterator<Item = TensorEntry>,
        metadata: Option<MetadataMap>,
    ) -> SaveResult<Self> {
        let tensors: Vec<TensorEntry> = tensors.into_iter().collect();
        if tensors.is_empty() {
            return Err(SaveError::InvalidInput(
                "cannot create checkpoint with empty tensor list".to_owned(),
            ));
        }
        Ok(Self { tensors, metadata })
    }

    /// Serialize the checkpoint into an owned byte buffer.
    pub fn to_bytes(&self) -> SaveResult<Vec<u8>> {
        let views = self.tensor_views()?;
        safetensors::serialize(views, self.metadata.clone()).map_err(SaveError::from)
    }

    fn tensor_views(&self) -> SaveResult<Vec<(&str, safetensors::tensor::TensorView<'_>)>> {
        self.tensors
            .iter()
            .map(|entry| {
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::from(entry.dtype()),
                    entry.shape().to_vec(),
                    entry.data(),
                )
                .map(|v| (entry.name(), v))
                .map_err(|e| SaveError::InvalidInput(e.to_string()))
            })
            .collect()
    }

    /// Discover `.safetensors` files in a directory.
    fn discover_files(path: impl AsRef<Path>) -> LoadResult<Vec<PathBuf>> {
        let path = path.as_ref();
        if path.is_file() {
            return Err(LoadError::InvalidMetadata(format!(
                "expected a directory of .safetensors files, got file path {}",
                path.display()
            )));
        }
        if !path.is_dir() {
            return Err(LoadError::InvalidMetadata(format!(
                "expected a directory of .safetensors files, path not found: {}",
                path.display()
            )));
        }

        let mut file_paths = Vec::new();
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
                file_paths.push(entry_path);
            }
        }

        file_paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        if file_paths.is_empty() {
            return Err(LoadError::InvalidMetadata(format!(
                "no .safetensors files found in {}",
                path.display()
            )));
        }

        Ok(file_paths)
    }
}

impl crate::formats::traits::Checkpoint for Checkpoint {
    type Model = Model;

    fn load(path: impl AsRef<Path>, backend: Backend) -> LoadResult<Self::Model> {
        let paths = Self::discover_files(path)?;
        let mut files = Vec::with_capacity(paths.len());

        match backend {
            Backend::Sync => {
                for path in paths {
                    let bytes = sync::File::open(&path)
                        .and_then(|file| file.read_all())
                        .map_err(LoadError::from)?;
                    files.push(FileData::parse(bytes)?);
                }
            }
            #[cfg(target_os = "linux")]
            Backend::IoUring => {
                for path in paths {
                    let bytes = fastio::uring::File::open(&path)
                        .and_then(|file| file.read_all())
                        .map_err(LoadError::from)?;
                    files.push(FileData::parse(bytes)?);
                }
            }
        }

        Model::from_files(files)
    }

    async fn aload(
        path: impl AsRef<Path> + Send,
        backend: AsyncBackend,
    ) -> LoadResult<Self::Model> {
        let paths = Self::discover_files(path)?;
        let mut files = Vec::with_capacity(paths.len());

        match backend {
            AsyncBackend::Tokio => {
                for path in paths {
                    let bytes = fastio_tokio::File::open(&path)
                        .await
                        .map_err(LoadError::from)?
                        .read_all()
                        .await
                        .map_err(LoadError::from)?;
                    files.push(FileData::parse(bytes)?);
                }
            }
        }

        Model::from_files(files)
    }

    fn open(path: impl AsRef<Path>) -> LoadResult<Self::Model> {
        let paths = Self::discover_files(path)?;
        let mut files = Vec::with_capacity(paths.len());

        for path in paths {
            let mmap = mmap::File::open(&path)
                .and_then(|file| file.map())
                .map_err(LoadError::from)?;
            files.push(FileData::parse(mmap)?);
        }

        Model::from_files(files)
    }

    fn save(&self, path: impl AsRef<Path>) -> SaveResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let views = self.tensor_views()?;
        safetensors::serialize_to_file(views, self.metadata.clone(), path).map_err(SaveError::from)
    }

    async fn asave(&self, path: impl AsRef<Path> + Send) -> SaveResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            tokio::fs::create_dir_all(parent).await?;
        }
        let bytes = self.to_bytes()?;
        let file = fastio_tokio::File::create(path)
            .await
            .map_err(SaveError::from)?;
        file.write_all_at(0, &bytes)
            .await
            .map_err(SaveError::from)?;
        file.sync_all().await.map_err(SaveError::from)
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::Checkpoint as SafeTensorsCheckpoint;
    use super::{Dtype, TensorEntry};
    use crate::formats::traits::{Checkpoint, Model};

    fn sample_checkpoint() -> SafeTensorsCheckpoint {
        SafeTensorsCheckpoint::new(
            [
                TensorEntry::new("a", vec![0u8; 4], vec![1usize], Dtype::F32).unwrap(),
                TensorEntry::new("b", vec![0u8; 16], vec![2usize], Dtype::F64).unwrap(),
            ],
            None,
        )
        .unwrap()
    }

    #[test]
    fn checkpoint_validates_empty_tensor_list() {
        let result = SafeTensorsCheckpoint::new(std::iter::empty::<TensorEntry>(), None);
        assert!(result.is_err());
    }

    #[test]
    fn validates_empty_name() {
        let entry = TensorEntry::new("", vec![0u8; 4], vec![1usize], Dtype::F32);
        assert!(entry.is_err());
    }

    #[test]
    fn validates_unknown_dtype() {
        let entry = TensorEntry::new("t", vec![0u8; 4], vec![1usize], Dtype::Unknown);
        assert!(entry.is_err());
    }

    #[test]
    fn checkpoint_roundtrip() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        sample_checkpoint().save(&path).unwrap();
        let model = SafeTensorsCheckpoint::load(dir.path(), crate::formats::Backend::Sync).unwrap();
        assert_eq!(model.len(), 2);
    }
}
