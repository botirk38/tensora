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
