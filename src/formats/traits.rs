//! Traits for tensor model access and serialization.
//!
//! These traits provide a uniform API across tensor formats.
//! Each format type implements the traits it supports.

use crate::formats::error::WriterResult;
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// Model Trait
// ============================================================================

/// Uniform read-only access to a loaded tensor model.
///
/// All format families implement this so callers can work with models
/// without knowing which format produced them.
pub trait Model {
    /// The tensor view type returned by this model.
    type Tensor<'a>: TensorView
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
    fn contains(&self, name: &str) -> bool;

    /// Returns tensor names as a borrowed slice (cached, sorted).
    fn tensor_names(&self) -> &[Arc<str>];

    /// Returns a view of the tensor with the given name, or `None` if missing.
    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>>;
}

// ============================================================================
// Tensor View Trait
// ============================================================================

/// Uniform read-only view of a single tensor's data.
///
/// Both format families implement this so callers can work with tensors
/// without knowing which format produced them.
pub trait TensorView {
    /// Shape in elements.
    fn shape(&self) -> &[usize];
    /// Dtype as a canonical string (e.g. `"float32"`, `"int64"`).
    fn dtype(&self) -> &str;
    /// Raw bytes of the tensor data.
    fn data(&self) -> &[u8];
}

// ============================================================================
// Serializer Traits
// ============================================================================

/// Trait for asynchronous format serializers.
///
/// Writes a complete model to disk using async I/O.
#[allow(async_fn_in_trait)]
pub trait AsyncSerializer {
    /// The input data type accepted by this serializer.
    type Input;

    /// Asynchronously serializes tensor data to the given path.
    async fn write(path: &Path, data: &Self::Input) -> WriterResult<()>;
}

/// Trait for synchronous format serializers.
///
/// Writes a complete model to disk using sync I/O.
pub trait SyncSerializer {
    /// The input data type accepted by this serializer.
    type Input;

    /// Synchronously serializes tensor data to the given path.
    fn write_sync(path: &Path, data: &Self::Input) -> WriterResult<()>;
}

#[cfg(test)]
mod tests {
    use super::{Model, TensorView};
    use std::sync::Arc;

    struct DummyTensor {
        shape: Vec<usize>,
        dtype: &'static str,
        data: Vec<u8>,
    }

    impl TensorView for DummyTensor {
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn dtype(&self) -> &str {
            self.dtype
        }
        fn data(&self) -> &[u8] {
            &self.data
        }
    }

    impl TensorView for &DummyTensor {
        fn shape(&self) -> &[usize] {
            (**self).shape()
        }
        fn dtype(&self) -> &str {
            (**self).dtype()
        }
        fn data(&self) -> &[u8] {
            (**self).data()
        }
    }

    struct DummyModel {
        names: Vec<Arc<str>>,
        tensors: Vec<DummyTensor>,
    }

    impl Model for DummyModel {
        type Tensor<'a>
            = &'a DummyTensor
        where
            Self: 'a;

        fn len(&self) -> usize {
            self.tensors.len()
        }

        fn contains(&self, name: &str) -> bool {
            self.names.iter().any(|n| n.as_ref() == name)
        }

        fn tensor_names(&self) -> &[Arc<str>] {
            &self.names
        }

        fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
            self.names
                .iter()
                .position(|n| n.as_ref() == name)
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
                    dtype: "f32",
                    data: vec![0; 8],
                },
                DummyTensor {
                    shape: vec![3],
                    dtype: "u8",
                    data: vec![1; 3],
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
    }
}
