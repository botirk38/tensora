//! `SafeTensors` format model.
//!
//! [`Model`] provides read-only access to tensors loaded from one or more
//! `.safetensors` files.  Loading is owned by
//! [`Checkpoint`](crate::formats::safetensors::Checkpoint).

use crate::formats::error::{LoadError, LoadResult};
use crate::formats::tensor::Dtype;
use fastio::{MmapRegion, OwnedBytes};
use safetensors::tensor::Metadata;
use std::collections::HashMap;
use std::sync::Arc;

use super::tensor::Tensor;

// ============================================================================
// Backing — storage abstraction
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

// ============================================================================
// FileData — one parsed SafeTensors file
// ============================================================================

#[derive(Debug, Clone)]
pub(crate) struct FileData {
    pub(crate) metadata: Metadata,
    pub(crate) data_start: usize,
    pub(crate) backing: Backing,
}

impl FileData {
    /// Parse a SafeTensors file from raw bytes or an mmap region.
    pub(crate) fn parse(backing: impl Into<Backing>) -> LoadResult<Self> {
        const N_LEN: usize = std::mem::size_of::<u64>();
        let backing = backing.into();
        let (header_json_len, metadata) =
            safetensors::SafeTensors::read_metadata(backing.as_slice())?;
        Ok(Self {
            metadata,
            data_start: header_json_len + N_LEN,
            backing,
        })
    }
}

// ============================================================================
// Model
// ============================================================================

/// SafeTensors model loaded from one or more `.safetensors` files.
///
/// Supports both eager (owned bytes) and lazy (mmap-backed) storage.
/// Single-file and multi-file models are represented uniformly.
#[derive(Debug, Clone)]
pub struct Model {
    /// All parsed SafeTensors files, in discovery order.
    files: Vec<FileData>,
    /// Maps tensor name → index into `files`.
    tensor_files: HashMap<Arc<str>, usize>,
    /// Sorted tensor names.
    tensor_names: Arc<[Arc<str>]>,
}

impl Model {
    /// Build a model from one or more parsed [`FileData`] objects.
    ///
    /// Used by [`Checkpoint`](crate::formats::safetensors::Checkpoint) after it
    /// owns the loading process.
    pub(crate) fn from_files(files: Vec<FileData>) -> LoadResult<Self> {
        let mut tensor_files: HashMap<Arc<str>, usize> = HashMap::new();
        let mut tensor_names: Vec<Arc<str>> = Vec::new();

        for (file_index, file) in files.iter().enumerate() {
            for name in file.metadata.offset_keys() {
                let name: Arc<str> = Arc::from(name.as_str());
                if tensor_files.insert(name.clone(), file_index).is_some() {
                    return Err(LoadError::InvalidMetadata(format!(
                        "duplicate tensor name across SafeTensors files: {name}"
                    )));
                }
                tensor_names.push(name);
            }
        }

        tensor_names.sort_unstable();

        Ok(Self {
            files,
            tensor_files,
            tensor_names: tensor_names.into(),
        })
    }

    /// Returns `true` if any file uses lazy (mmap-backed) storage.
    #[must_use]
    pub fn is_lazy(&self) -> bool {
        self.files
            .iter()
            .any(|f| matches!(f.backing, Backing::Mmap(_)))
    }
}

fn arc_to_str(arc: &Arc<str>) -> &str {
    arc.as_ref()
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
        self.tensor_names.len()
    }

    fn tensor_names(&self) -> Self::Names<'_> {
        self.tensor_names.iter().map(arc_to_str)
    }

    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
        let file = &self.files[*self.tensor_files.get(name)?];
        let info = file.metadata.info(name)?;
        let (start, end) = info.data_offsets;
        let data = file.backing.as_slice()[file.data_start..].get(start..end)?;
        Some(Tensor::new(&info.shape, Dtype::from(info.dtype), data))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::{FileData, Model as SafeTensorsModel, Tensor as SafeTensorsTensor};
    use crate::formats::tensor::Dtype;
    use crate::formats::traits::{Model, Tensor};
    use fastio::OwnedBytes;

    fn minimal_file_data() -> FileData {
        // Minimal valid SafeTensors: 8-byte LE u64 length + header JSON "{}"
        let header = b"{}";
        let len = header.len() as u64;
        let mut data = Vec::with_capacity(8 + header.len());
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(header);
        FileData::parse(OwnedBytes::from_vec(data)).expect("parse minimal file")
    }

    #[test]
    fn model_from_single_file() {
        let model =
            SafeTensorsModel::from_files(vec![minimal_file_data()]).expect("single file ok");
        assert!(model.is_empty());
        assert!(model.tensor_names().next().is_none());
    }

    #[test]
    fn model_from_multiple_files() {
        let f1 = minimal_file_data();
        let f2 = minimal_file_data();
        let model = SafeTensorsModel::from_files(vec![f1, f2]).expect("two files ok");
        assert!(model.is_empty());
    }

    #[test]
    fn model_len_is_zero_for_empty_file() {
        let model = SafeTensorsModel::from_files(vec![minimal_file_data()]).expect("parse");
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn model_is_empty_when_no_tensors() {
        let model = SafeTensorsModel::from_files(vec![minimal_file_data()]).expect("parse");
        assert!(model.is_empty());
    }

    #[test]
    fn tensor_returns_none_for_missing() {
        let model = SafeTensorsModel::from_files(vec![minimal_file_data()]).expect("parse");
        assert!(model.tensor("nonexistent").is_none());
    }

    #[test]
    fn model_contains_uses_tensor_lookup() {
        let model = SafeTensorsModel::from_files(vec![minimal_file_data()]).expect("parse");
        assert!(!model.contains("any"));
    }

    #[test]
    fn tensor_stride_returns_none() {
        // SafeTensors tensors do not store explicit stride
        let shape = vec![2, 3, 4];
        let data = vec![0u8; 2 * 3 * 4 * 4]; // f32
        let tensor = SafeTensorsTensor::new(&shape, Dtype::F32, &data);
        assert_eq!(tensor.stride(), None);
    }
}
