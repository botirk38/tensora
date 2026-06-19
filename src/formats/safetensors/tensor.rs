//! Tensor view type for the SafeTensors format.

use crate::formats::tensor::Dtype;

/// A tensor view into a SafeTensors model.
///
/// Borrows shape and data directly from the loaded shard buffer — zero-copy.
/// SafeTensors tensors are always contiguous, so stride is not stored.
#[derive(Debug)]
pub struct Tensor<'a> {
    pub(crate) shape: &'a [usize],
    pub(crate) dtype: Dtype,
    pub(crate) data: &'a [u8],
}

impl<'a> Tensor<'a> {
    #[inline]
    #[must_use]
    pub(crate) fn new(shape: &'a [usize], dtype: Dtype, data: &'a [u8]) -> Self {
        Self { shape, dtype, data }
    }

    /// Shape in elements per dimension.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Element type.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Raw tensor bytes.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.data
    }
}

impl crate::formats::traits::Tensor for Tensor<'_> {
    #[inline]
    fn shape(&self) -> &[usize] { self.shape }
    #[inline]
    fn dtype(&self) -> Dtype { self.dtype }
    #[inline]
    fn data(&self) -> &[u8] { self.data }
    #[inline]
    fn stride(&self) -> Option<&[usize]> {
        // SafeTensors are always contiguous; stride is not stored
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::traits::Tensor as TensorTrait;

    #[test]
    fn accessors() {
        let shape = [2usize, 3];
        let data = [0u8; 24];
        let t = Tensor::new(&shape, Dtype::F32, &data);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), Dtype::F32);
        assert_eq!(t.data().len(), 24);
        assert_eq!(t.stride(), None);
    }
}
