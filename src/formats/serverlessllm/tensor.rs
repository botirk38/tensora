//! Tensor types for the ServerlessLLM format.
//!
//! [`TensorEntry`] is the metadata record stored in the index — shape, stride,
//! dtype, byte offset, size, and partition ID. [`Tensor`] is the live view
//! produced when a model is loaded, borrowing metadata from a [`TensorEntry`]
//! and holding the actual bytes.

use crate::formats::error::SaveError;
use crate::formats::tensor::{Dtype, TensorMeta};
use crate::io::buffer::MmapRegion;
use std::sync::Arc;

use super::ids::PartitionId;

// ============================================================================
// TensorEntry — metadata record (replaces PartitionedTensor)
// ============================================================================

/// Metadata record for a single tensor in a ServerlessLLM checkpoint.
///
/// Stores everything needed to locate and describe the tensor: where it sits
/// in its partition file (`offset`, `size`), its logical layout (`shape`,
/// `stride`, `dtype`), and which partition holds its bytes (`partition_id`).
///
/// Construct via [`TensorEntry::from_parts`] (raw fields) or
/// [`TensorEntry::new`] (from an existing [`TensorMeta`]).
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub(crate) meta: TensorMeta,
    pub(crate) partition_id: PartitionId,
}

impl TensorEntry {
    /// Build from an already-validated [`TensorMeta`] and a partition ID.
    #[inline]
    #[must_use]
    pub fn new(meta: TensorMeta, partition_id: PartitionId) -> Self {
        Self { meta, partition_id }
    }

    /// Build from raw components, running [`TensorMeta`] validation.
    ///
    /// # Errors
    ///
    /// Returns `SaveError::InvalidInput` if `dtype` is [`Dtype::Unknown`],
    /// `shape.len() != stride.len()`, or `offset + size` overflows `u64`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        offset: u64,
        size: u64,
        shape: impl Into<Arc<[usize]>>,
        stride: impl Into<Arc<[usize]>>,
        dtype: Dtype,
        partition_id: PartitionId,
    ) -> Result<Self, SaveError> {
        Ok(Self { meta: TensorMeta::new(offset, size, shape, stride, dtype)?, partition_id })
    }

    /// Returns the underlying format-agnostic metadata.
    #[inline] #[must_use]
    pub fn meta(&self) -> &TensorMeta { &self.meta }

    /// Byte offset within the partition file.
    #[inline] #[must_use]
    pub fn offset(&self) -> u64 { self.meta.offset }

    /// Size in bytes.
    #[inline] #[must_use]
    pub fn size(&self) -> u64 { self.meta.size }

    /// Shape in elements per dimension.
    #[inline] #[must_use]
    pub fn shape(&self) -> &[usize] { self.meta.shape() }

    /// Stride in elements per dimension.
    #[inline] #[must_use]
    pub fn stride(&self) -> &[usize] { self.meta.stride() }

    /// Element type.
    #[inline] #[must_use]
    pub fn dtype(&self) -> Dtype { self.meta.dtype() }

    /// Partition this tensor lives in.
    #[inline] #[must_use]
    pub fn partition_id(&self) -> PartitionId { self.partition_id }
}

// ============================================================================
// Tensor — live view
// ============================================================================

/// A live tensor view into a loaded ServerlessLLM model.
///
/// Borrows its metadata from the model's index ([`TensorEntry`]) and holds
/// either a slice into an eagerly-loaded buffer or a memory-mapped sub-region.
#[derive(Debug)]
pub struct Tensor<'a> {
    meta: &'a TensorMeta,
    data: TensorData<'a>,
}

#[derive(Debug)]
enum TensorData<'a> {
    Eager(&'a [u8]),
    Mmap(MmapRegion),
}

impl<'a> Tensor<'a> {
    pub(crate) fn eager(entry: &'a TensorEntry, data: &'a [u8]) -> Self {
        Self { meta: &entry.meta, data: TensorData::Eager(data) }
    }

    pub(crate) fn mmap(entry: &'a TensorEntry, data: MmapRegion) -> Self {
        Self { meta: &entry.meta, data: TensorData::Mmap(data) }
    }

    /// Shape in elements per dimension.
    #[inline] #[must_use]
    pub fn shape(&self) -> &[usize] { self.meta.shape() }

    /// Element type.
    #[inline] #[must_use]
    pub fn dtype(&self) -> Dtype { self.meta.dtype() }

    /// Raw tensor bytes.
    #[inline] #[must_use]
    pub fn data(&self) -> &[u8] {
        match &self.data {
            TensorData::Eager(b) => b,
            TensorData::Mmap(r) => r.as_slice(),
        }
    }
}

impl crate::formats::traits::Tensor for Tensor<'_> {
    #[inline] fn shape(&self) -> &[usize] { self.shape() }
    #[inline] fn dtype(&self) -> Dtype { self.dtype() }
    #[inline] fn data(&self) -> &[u8] { self.data() }
    #[inline] fn stride(&self) -> Option<&[usize]> { Some(self.meta.stride()) }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::traits::Tensor as TensorTrait;

    fn entry() -> TensorEntry {
        TensorEntry::from_parts(0, 4, vec![2usize, 2], vec![2usize, 1], Dtype::F32, PartitionId::new(0)).unwrap()
    }

    #[test]
    fn entry_accessors() {
        let e = TensorEntry::from_parts(8, 16, vec![4usize], vec![1usize], Dtype::F64, PartitionId::new(2)).unwrap();
        assert_eq!(e.offset(), 8);
        assert_eq!(e.size(), 16);
        assert_eq!(e.shape(), &[4]);
        assert_eq!(e.stride(), &[1]);
        assert_eq!(e.dtype(), Dtype::F64);
        assert_eq!(e.partition_id(), PartitionId::new(2));
    }

    #[test]
    fn entry_from_meta() {
        let meta = TensorMeta::new(0, 4, vec![1usize], vec![1usize], Dtype::U8).unwrap();
        let e = TensorEntry::new(meta, PartitionId::new(1));
        assert_eq!(e.partition_id(), PartitionId::new(1));
        assert_eq!(e.dtype(), Dtype::U8);
    }

    #[test]
    fn entry_rejects_unknown_dtype() {
        assert!(TensorEntry::from_parts(0, 4, vec![1usize], vec![1usize], Dtype::Unknown, PartitionId::new(0)).is_err());
    }

    #[test]
    fn tensor_eager_access() {
        let e = entry();
        let data: Arc<[u8]> = Arc::from(vec![1u8, 2, 3, 4]);
        let t = Tensor::eager(&e, &data);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.dtype(), Dtype::F32);
        assert_eq!(t.data(), &[1, 2, 3, 4]);
        assert_eq!(t.stride(), Some(&[2usize, 1][..]));
    }
}
