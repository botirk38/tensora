//! [`Model`] trait — uniform read-only access to a loaded tensor model.

use super::Tensor;

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

    fn dummy_model() -> DummyModel {
        DummyModel {
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
        }
    }

    #[test]
    fn len_and_contains() {
        let m = dummy_model();
        assert_eq!(m.len(), 2);
        assert!(!m.is_empty());
        assert!(m.contains("a"));
        assert!(!m.contains("c"));
    }

    #[test]
    fn tensor_names_exact_size() {
        assert_eq!(dummy_model().tensor_names().len(), 2);
    }

    #[test]
    fn tensor_lookup() {
        let m = dummy_model();
        assert!(m.tensor("a").is_some());
        assert!(m.tensor("c").is_none());
        assert_eq!(m.tensor("a").unwrap().shape(), &[2]);
        assert_eq!(m.tensor("a").unwrap().stride(), Some([4usize].as_slice()));
    }

    #[test]
    fn empty_model() {
        let m = DummyModel {
            names: vec![],
            tensors: vec![],
        };
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn missing_tensor_returns_none() {
        let m = DummyModel {
            names: vec!["x".into()],
            tensors: vec![DummyTensor {
                shape: vec![1],
                dtype: Dtype::F32,
                data: vec![0; 4],
                stride: vec![4],
            }],
        };
        assert!(m.tensor("nonexistent").is_none());
        assert!(!m.contains("nonexistent"));
    }
}
