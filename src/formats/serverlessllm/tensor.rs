//! Tensor type for the ServerlessLLM format.

use crate::formats::tensor::Dtype;
use crate::io::buffer::MmapRegion;

use super::index::TensorDescriptor;

/// A tensor view into a ServerlessLLM model.
///
/// The tensor borrows metadata from the model's index and either borrows bytes
/// from an eager partition buffer or holds a sub-region of an mmap partition.
#[derive(Debug)]
pub struct Tensor<'a> {
    desc: &'a TensorDescriptor,
    data: TensorData<'a>,
}

#[derive(Debug)]
enum TensorData<'a> {
    Eager(&'a [u8]),
    Mmap(MmapRegion),
}

impl<'a> Tensor<'a> {
    pub(crate) fn eager(desc: &'a TensorDescriptor, data: &'a [u8]) -> Self {
        Self {
            desc,
            data: TensorData::Eager(data),
        }
    }

    pub(crate) fn mmap(desc: &'a TensorDescriptor, data: MmapRegion) -> Self {
        Self {
            desc,
            data: TensorData::Mmap(data),
        }
    }

    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.desc.shape()
    }

    #[inline]
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.desc.dtype()
    }

    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        match &self.data {
            TensorData::Eager(data) => data,
            TensorData::Mmap(region) => region.as_slice(),
        }
    }
}

impl crate::formats::traits::Tensor for Tensor<'_> {
    #[inline]
    fn shape(&self) -> &[usize] {
        self.shape()
    }

    #[inline]
    fn dtype(&self) -> Dtype {
        self.dtype()
    }

    #[inline]
    fn data(&self) -> &[u8] {
        self.data()
    }

    #[inline]
    fn stride(&self) -> Option<&[usize]> {
        Some(self.desc.stride())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::serverlessllm::ids::PartitionId;
    use crate::formats::serverlessllm::index::TensorDescriptor;
    use crate::formats::traits::Tensor as TensorTrait;
    use std::sync::Arc;

    fn make_desc(dtype: Dtype) -> TensorDescriptor {
        // Use the public constructor via Index
        let json = br#"{"test": [0, 4, [2, 2], [2, 1], "f32", 0]}"#;
        let index = crate::formats::serverlessllm::Index::from_bytes(json).unwrap();
        index.get("test").unwrap().clone()
    }

    #[test]
    fn eager_tensor_access() {
        let desc = make_desc(Dtype::F32);
        let data: Arc<[u8]> = Arc::from(vec![1, 2, 3, 4]);
        let tensor = Tensor::eager(&desc, &data);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), Dtype::F32);
        assert_eq!(tensor.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn tensor_stride_returns_descriptor_stride() {
        let desc = make_desc(Dtype::F32);
        let data: Arc<[u8]> = Arc::from(vec![1, 2, 3, 4]);
        let tensor = Tensor::eager(&desc, &data);
        let expected: &[usize] = &[2, 1];
        assert_eq!(tensor.stride(), Some(expected));
    }
}
