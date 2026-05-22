//! Tensor data containers for ServerlessLLM format.

use std::sync::Arc;

use crate::backends;
use crate::formats::traits::TensorView;

use super::index::TensorDescriptor;

/// Owned tensor with shared backing buffer.
/// Multiple tensors from the same partition share the same buffer via Arc.
#[derive(Debug, Clone)]
pub struct Tensor {
    backing: Arc<[u8]>,
    desc: Arc<TensorDescriptor>,
}

impl Tensor {
    /// Creates a new Tensor from shared backing and descriptor.
    #[inline]
    #[must_use]
    pub fn from_shared(backing: Arc<[u8]>, desc: Arc<TensorDescriptor>) -> Self {
        Self { backing, desc }
    }

    /// Returns the raw tensor data as a slice (zero-copy).
    /// Slices from the shared backing buffer using the descriptor's offset.
    /// Panics if offset + size overflows or exceeds buffer bounds.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        let start = usize::try_from(self.desc.offset).expect("offset overflow");
        let end = start
            .checked_add(self.desc.size)
            .expect("offset + size overflow");
        &self.backing[start..end]
    }

    /// Returns the tensor's data type.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.desc.dtype
    }

    /// Returns the tensor's shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.desc.shape
    }

    /// Returns the tensor's stride.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.desc.stride
    }

    /// Returns the tensor's size in bytes.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.desc.size
    }

    /// Returns the partition id containing this tensor.
    #[inline]
    #[must_use]
    pub fn partition_id(&self) -> usize {
        self.desc.partition_id
    }
}

impl TensorView for Tensor {
    #[inline]
    fn shape(&self) -> &[usize] {
        self.shape()
    }

    #[inline]
    fn dtype(&self) -> &str {
        self.dtype()
    }

    #[inline]
    fn data(&self) -> &[u8] {
        self.data()
    }
}

impl TensorView for &Tensor {
    #[inline]
    fn shape(&self) -> &[usize] {
        (**self).shape()
    }

    #[inline]
    fn dtype(&self) -> &str {
        (**self).dtype()
    }

    #[inline]
    fn data(&self) -> &[u8] {
        (**self).data()
    }
}

/// View into a memory-mapped tensor with metadata access (lazy loading).
#[derive(Debug)]
pub struct TensorMmap {
    mmap: backends::mmap::Mmap,
    desc: Arc<TensorDescriptor>,
}

impl TensorMmap {
    /// Creates a new TensorMmap from memory-mapped data.
    #[inline]
    #[must_use]
    pub fn new(mmap: backends::mmap::Mmap, desc: Arc<TensorDescriptor>) -> Self {
        Self { mmap, desc }
    }

    /// Returns the memory-mapped tensor data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.mmap.as_slice()
    }

    /// Returns the tensor's data type.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.desc.dtype
    }

    /// Returns the tensor's shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.desc.shape
    }

    /// Returns the tensor's stride.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.desc.stride
    }

    /// Returns the tensor's size in bytes.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.desc.size
    }
}

impl TensorView for TensorMmap {
    #[inline]
    fn shape(&self) -> &[usize] {
        self.shape()
    }

    #[inline]
    fn dtype(&self) -> &str {
        self.dtype()
    }

    #[inline]
    fn data(&self) -> &[u8] {
        self.data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_desc(offset: u64, size: usize, partition_id: usize) -> Arc<TensorDescriptor> {
        Arc::new(TensorDescriptor {
            offset,
            size,
            shape: vec![2, 4].into(),
            stride: vec![4, 1].into(),
            dtype: "torch.float32".into(),
            partition_id,
        })
    }

    #[test]
    fn tensor_from_shared_data_access() {
        let backing: Arc<[u8]> = Arc::from(vec![10u8, 20, 30, 40, 50, 60, 70, 80]);
        let desc = make_desc(2, 4, 0);
        let t = Tensor::from_shared(backing, desc);
        assert_eq!(t.data(), &[30, 40, 50, 60]);
    }

    #[test]
    fn tensor_shape_dtype_stride_size() {
        let backing: Arc<[u8]> = Arc::from(vec![0u8; 32]);
        let desc = make_desc(0, 32, 1);
        let t = Tensor::from_shared(backing, desc);
        assert_eq!(t.shape(), &[2, 4]);
        assert_eq!(t.dtype(), "torch.float32");
        assert_eq!(t.stride(), &[4, 1]);
        assert_eq!(t.size(), 32);
    }

    #[test]
    fn tensor_partition_id() {
        let backing: Arc<[u8]> = Arc::from(vec![0u8; 8]);
        let desc = make_desc(0, 8, 42);
        let t = Tensor::from_shared(backing, desc);
        assert_eq!(t.partition_id(), 42);
    }

    #[test]
    fn tensor_view_impl() {
        let backing: Arc<[u8]> = Arc::from(vec![1u8, 2, 3, 4]);
        let desc = make_desc(0, 4, 0);
        let t = Tensor::from_shared(backing, desc);
        let tv: &dyn TensorView = &t;
        assert_eq!(tv.shape(), &[2, 4]);
        assert_eq!(tv.dtype(), "torch.float32");
        assert_eq!(tv.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn tensor_ref_view_impl() {
        let backing: Arc<[u8]> = Arc::from(vec![5u8, 6, 7, 8]);
        let desc = make_desc(0, 4, 0);
        let t = Tensor::from_shared(backing, desc);
        let r = &t;
        let tv: &dyn TensorView = &r;
        assert_eq!(tv.data(), &[5, 6, 7, 8]);
    }

    #[test]
    fn tensor_data_boundary() {
        let backing: Arc<[u8]> = Arc::from(vec![0u8, 0, 0, 0, 1, 2, 3, 4, 0, 0]);
        let desc = make_desc(4, 4, 0);
        let t = Tensor::from_shared(backing, desc);
        assert_eq!(t.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn tensor_clone() {
        let backing: Arc<[u8]> = Arc::from(vec![1u8; 4]);
        let desc = make_desc(0, 4, 0);
        let t = Tensor::from_shared(backing, desc);
        let t2 = t.clone();
        assert_eq!(t.data(), t2.data());
    }
}
