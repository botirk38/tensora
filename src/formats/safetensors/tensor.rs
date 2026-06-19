//! Tensor types for the SafeTensors format.
//!
//! [`TensorEntry`] is the write-side record supplied when constructing a
//! [`Checkpoint`](crate::formats::safetensors::Checkpoint) — it carries the
//! name, raw bytes, shape, and dtype of one tensor.
//!
//! [`Tensor`] is the read-side view produced when a model is queried.  It
//! borrows data directly from the loaded file buffer — zero-copy.

use crate::formats::error::{SaveError, SaveResult};
use crate::formats::tensor::Dtype;

// ============================================================================
// TensorEntry — write-side record
// ============================================================================

/// A single tensor to be written into a SafeTensors checkpoint.
///
/// Construct via [`TensorEntry::new`], which validates the name and dtype.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub(crate) name: String,
    pub(crate) data: Vec<u8>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: Dtype,
}

impl TensorEntry {
    /// Create a new tensor entry.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::InvalidInput`] if:
    /// - `name` is empty
    /// - `dtype` is [`Dtype::Unknown`]
    pub fn new(
        name: impl Into<String>,
        data: impl Into<Vec<u8>>,
        shape: impl Into<Vec<usize>>,
        dtype: Dtype,
    ) -> SaveResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(SaveError::InvalidInput(
                "tensor name cannot be empty".to_owned(),
            ));
        }
        if dtype == Dtype::Unknown {
            return Err(SaveError::InvalidInput(
                "cannot use Unknown dtype for tensor".to_owned(),
            ));
        }
        Ok(Self {
            name,
            data: data.into(),
            shape: shape.into(),
            dtype,
        })
    }

    /// Tensor name.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Raw tensor bytes.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Shape in elements per dimension.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element type.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
}

// ============================================================================
// Tensor — read-side view
// ============================================================================

/// A tensor view into a SafeTensors model.
///
/// Borrows shape and data directly from the loaded file buffer — zero-copy.
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
    fn shape(&self) -> &[usize] {
        self.shape
    }
    #[inline]
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    #[inline]
    fn data(&self) -> &[u8] {
        self.data
    }
    #[inline]
    fn stride(&self) -> Option<&[usize]> {
        // SafeTensors are always contiguous; stride is not stored
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::traits::Tensor as TensorTrait;

    // ---- TensorEntry -------------------------------------------------------

    #[test]
    fn tensor_entry_valid() {
        let entry = TensorEntry::new("weight", vec![0u8; 4], vec![1usize], Dtype::F32).unwrap();
        assert_eq!(entry.name(), "weight");
        assert_eq!(entry.data().len(), 4);
        assert_eq!(entry.shape(), &[1]);
        assert_eq!(entry.dtype(), Dtype::F32);
    }

    #[test]
    fn tensor_entry_rejects_empty_name() {
        assert!(TensorEntry::new("", vec![0u8; 4], vec![1usize], Dtype::F32).is_err());
    }

    #[test]
    fn tensor_entry_rejects_unknown_dtype() {
        assert!(TensorEntry::new("x", vec![0u8; 4], vec![1usize], Dtype::Unknown).is_err());
    }

    // ---- Tensor ------------------------------------------------------------

    #[test]
    fn tensor_accessors() {
        let shape = [2usize, 3];
        let data = [0u8; 24];
        let t = Tensor::new(&shape, Dtype::F32, &data);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), Dtype::F32);
        assert_eq!(t.data().len(), 24);
        assert_eq!(t.stride(), None);
    }
}
