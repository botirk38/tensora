//! Traits for tensor model access and checkpoint serialization.
//!
//! These traits provide a uniform API across tensor formats.
//! Each format type implements the traits it supports.

use crate::formats::error::{LoadResult, SaveResult};
use crate::formats::tensor::Dtype;
use crate::formats::{AsyncBackend, Backend};
use std::future::Future;
use std::path::Path;

// ============================================================================
// Model Trait
// ============================================================================

/// Uniform read-only access to a loaded tensor model.
///
/// All format families implement this so callers can work with models
/// without knowing which format produced them.
pub trait Model {
    /// The tensor type returned by this model.
    type Tensor<'a>: Tensor
    where
        Self: 'a;

    /// Iterator type for tensor names.
    type Names<'a>: ExactSizeIterator<Item = &'a str>
    where
        Self: 'a;

    /// Returns the number of tensors.
    fn len(&self) -> usize;

    /// Returns true if there are no tensors.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if a tensor with the given name exists.
    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.tensor(name).is_some()
    }

    /// Returns an iterator over tensor names.
    fn tensor_names(&self) -> Self::Names<'_>;

    /// Returns the tensor with the given name, or `None` if missing.
    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>>;
}

// ============================================================================
// Tensor Trait
// ============================================================================

/// Uniform read-only access to a tensor's data.
///
/// Each format implements this on its own concrete tensor type. A tensor is
/// a value object carrying shape, dtype, and contiguous bytes.
pub trait Tensor {
    /// Shape in elements.
    fn shape(&self) -> &[usize];
    /// Dtype as a canonical value.
    fn dtype(&self) -> Dtype;
    /// Raw bytes of the tensor data.
    fn data(&self) -> &[u8];
    /// Stride in bytes per dimension, if available.
    fn stride(&self) -> Option<&[usize]> {
        None
    }
}

// ============================================================================
// Checkpoint Trait
// ============================================================================

/// Trait for loading and saving a format-specific checkpoint.
///
/// Implement this on format checkpoint types to provide loading (`load`,
/// `aload`, `open`) and saving (`save`, `asave`) paths. Loading produces
/// the format's [`Model`] type; saving consumes a checkpoint instance.
/// Sync and async variants should be semantically equivalent.
pub trait Checkpoint {
    /// The model type produced by this checkpoint.
    type Model: Model;

    /// Load a checkpoint eagerly using the chosen blocking backend.
    fn load(path: impl AsRef<Path>, backend: Backend) -> LoadResult<Self::Model>;

    /// Load a checkpoint eagerly using the chosen async backend.
    fn aload(
        path: impl AsRef<Path> + Send,
        backend: AsyncBackend,
    ) -> impl Future<Output = LoadResult<Self::Model>> + Send;

    /// Open a checkpoint lazily (e.g. via memory mapping).
    fn open(path: impl AsRef<Path>) -> LoadResult<Self::Model>;

    /// Write the checkpoint to `path` synchronously.
    fn save(&self, path: impl AsRef<Path>) -> SaveResult<()>;

    /// Write the checkpoint to `path` asynchronously.
    fn asave(
        &self,
        path: impl AsRef<Path> + Send,
    ) -> impl Future<Output = SaveResult<()>> + Send;
}

#[cfg(test)]
mod tests {
    use super::{Model, Tensor as TensorTrait};
    use crate::formats::tensor::Dtype;

    struct DummyTensor {
        shape: Vec<usize>,
        dtype: Dtype,
        data: Vec<u8>,
        stride: Vec<usize>,
    }

    impl TensorTrait for DummyTensor {
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn dtype(&self) -> Dtype {
            self.dtype
        }
        fn data(&self) -> &[u8] {
            &self.data
        }
        fn stride(&self) -> Option<&[usize]> {
            Some(&self.stride)
        }
    }

    impl TensorTrait for &DummyTensor {
        fn shape(&self) -> &[usize] {
            (**self).shape()
        }
        fn dtype(&self) -> Dtype {
            (**self).dtype()
        }
        fn data(&self) -> &[u8] {
            (**self).data()
        }
        fn stride(&self) -> Option<&[usize]> {
            (**self).stride()
        }
    }

    struct DummyModel {
        names: Vec<String>,
        tensors: Vec<DummyTensor>,
    }

    impl DummyModel {
        fn names_iter(&self) -> impl ExactSizeIterator<Item = &str> {
            self.names.iter().map(|s| s.as_str())
        }
    }

    impl Model for DummyModel {
        type Tensor<'a>
            = &'a DummyTensor
        where
            Self: 'a;
        type Names<'a>
            = std::iter::Map<std::slice::Iter<'a, String>, fn(&'a String) -> &'a str>
        where
            Self: 'a;

        fn len(&self) -> usize {
            self.tensors.len()
        }

        fn tensor_names(&self) -> Self::Names<'_> {
            self.names.iter().map(|s| s.as_str())
        }

        fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
            self.names
                .iter()
                .position(|n| n == name)
                .map(|i| &self.tensors[i])
        }
    }

    #[test]
    fn model_trait_works() {
        let model = DummyModel {
            names: vec!["a".into(), "b".into()],
            tensors: vec![
                DummyTensor {
                    shape: vec![2],
                    dtype: Dtype::F32,
                    data: vec![0; 8],
                    stride: vec![4],
                },
                DummyTensor {
                    shape: vec![3],
                    dtype: Dtype::U8,
                    data: vec![1; 3],
                    stride: vec![1],
                },
            ],
        };
        assert_eq!(model.len(), 2);
        assert!(!model.is_empty());
        assert!(model.contains("a"));
        assert!(!model.contains("c"));
        assert_eq!(model.tensor_names().len(), 2);
        assert!(model.tensor("a").is_some());
        assert!(model.tensor("c").is_none());
        assert_eq!(model.tensor("a").unwrap().shape(), &[2]);
        let expected_stride: &[usize] = &[4];
        assert_eq!(model.tensor("a").unwrap().stride(), Some(expected_stride));
    }

    #[test]
    fn model_is_empty_when_empty() {
        let model = DummyModel {
            names: vec![],
            tensors: vec![],
        };
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn model_tensor_returns_none_for_missing() {
        let model = DummyModel {
            names: vec!["x".into()],
            tensors: vec![DummyTensor {
                shape: vec![1],
                dtype: Dtype::F32,
                data: vec![0; 4],
                stride: vec![4],
            }],
        };
        assert!(model.tensor("nonexistent").is_none());
        assert!(!model.contains("nonexistent"));
    }

    #[test]
    fn tensor_data_access() {
        let t = DummyTensor {
            shape: vec![2, 3],
            dtype: Dtype::F16,
            data: vec![1, 2, 3, 4, 5, 6],
            stride: vec![12, 4],
        };
        let view: &dyn TensorTrait = &t;
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.dtype(), Dtype::F16);
        assert_eq!(view.data(), &[1, 2, 3, 4, 5, 6]);
        let expected_stride: &[usize] = &[12, 4];
        assert_eq!(view.stride(), Some(expected_stride));
    }

    #[test]
    fn tensor_stride_default_is_none() {
        struct NoStrideTensor;
        impl TensorTrait for NoStrideTensor {
            fn shape(&self) -> &[usize] {
                &[2, 2]
            }
            fn dtype(&self) -> Dtype {
                Dtype::F32
            }
            fn data(&self) -> &[u8] {
                &[0; 16]
            }
            // No override of stride()
        }
        let t = NoStrideTensor;
        assert_eq!(t.stride(), None);
    }
}
