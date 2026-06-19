//! [`Tensor`] trait — uniform read-only access to a single tensor's data.

use crate::formats::tensor::Dtype;

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

#[cfg(test)]
mod tests {
    use super::Tensor as TensorTrait;
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

    #[test]
    fn data_access() {
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
        assert_eq!(view.stride(), Some([12usize, 4].as_slice()));
    }

    #[test]
    fn stride_default_is_none() {
        struct Bare;
        impl TensorTrait for Bare {
            fn shape(&self) -> &[usize] {
                &[2, 2]
            }
            fn dtype(&self) -> Dtype {
                Dtype::F32
            }
            fn data(&self) -> &[u8] {
                &[0; 16]
            }
        }
        assert_eq!(Bare.stride(), None);
    }
}
