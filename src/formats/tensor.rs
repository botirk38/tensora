//! Shared tensor primitives across all formats.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::formats::error::SaveError;

/// Canonical tensor data type.
///
/// Used by the common [`Tensor`](crate::formats::traits::Tensor) trait and by
/// format-specific tensor implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// Boolean.
    Bool,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 16-bit integer.
    U16,
    /// IEEE 754 half-precision float.
    F16,
    /// IEEE 754 single-precision float.
    F32,
    /// IEEE 754 double-precision float.
    F64,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// Brain floating-point 16-bit.
    Bf16,
    /// Unknown or unsupported dtype.
    Unknown,
}

impl Dtype {
    /// Returns the dtype as a normalized string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Bf16 => "bf16",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Dtype {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bool" | "b" => Ok(Self::Bool),
            "u8" => Ok(Self::U8),
            "i8" => Ok(Self::I8),
            "i16" => Ok(Self::I16),
            "u16" => Ok(Self::U16),
            "f16" => Ok(Self::F16),
            "f32" | "f" => Ok(Self::F32),
            "f64" | "d" => Ok(Self::F64),
            "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64),
            "u32" => Ok(Self::U32),
            "u64" => Ok(Self::U64),
            "bf16" => Ok(Self::Bf16),
            other => Err(format!("unknown dtype: {other}")),
        }
    }
}

impl From<safetensors::Dtype> for Dtype {
    fn from(dtype: safetensors::Dtype) -> Self {
        match dtype {
            safetensors::Dtype::BOOL => Self::Bool,
            safetensors::Dtype::U8 => Self::U8,
            safetensors::Dtype::I8 => Self::I8,
            safetensors::Dtype::I16 => Self::I16,
            safetensors::Dtype::U16 => Self::U16,
            safetensors::Dtype::F16 => Self::F16,
            safetensors::Dtype::F32 => Self::F32,
            safetensors::Dtype::F64 => Self::F64,
            safetensors::Dtype::I32 => Self::I32,
            safetensors::Dtype::I64 => Self::I64,
            safetensors::Dtype::U32 => Self::U32,
            safetensors::Dtype::U64 => Self::U64,
            safetensors::Dtype::BF16 => Self::Bf16,
            _ => Self::Unknown,
        }
    }
}

// ============================================================================
// TensorMeta
// ============================================================================

/// Format-agnostic tensor metadata.
///
/// Carries everything needed to describe a tensor's layout in a checkpoint
/// file: where it starts (`offset`), how large it is in bytes (`size`),
/// its logical shape and stride, and its element type. Intentionally contains
/// no format-specific fields such as partition IDs — those are added by
/// format-specific wrappers (e.g. `serverlessllm::TensorEntry`).
///
/// # Construction
///
/// Use [`TensorMeta::new`], which validates all invariants on creation.
///
/// # Example
///
/// ```rust
/// use tensora::formats::tensor::{Dtype, TensorMeta};
///
/// let meta = TensorMeta::new(0, 16, vec![2usize, 2], vec![2usize, 1], Dtype::F32).unwrap();
/// assert_eq!(meta.shape(), &[2, 2]);
/// assert_eq!(meta.stride(), &[2, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Byte offset within the storage region.
    pub(crate) offset: u64,
    /// Size in bytes.
    pub(crate) size: u64,
    /// Shape in elements per dimension.
    pub(crate) shape: Arc<[usize]>,
    /// Stride in elements per dimension.
    pub(crate) stride: Arc<[usize]>,
    /// Element type.
    pub(crate) dtype: Dtype,
}

impl TensorMeta {
    /// Create a new validated `TensorMeta`.
    ///
    /// # Errors
    ///
    /// Returns `SaveError::InvalidInput` if:
    /// - `dtype` is [`Dtype::Unknown`]
    /// - `shape.len() != stride.len()`
    /// - `offset + size` would overflow `u64`
    pub fn new(
        offset: u64,
        size: u64,
        shape: impl Into<Arc<[usize]>>,
        stride: impl Into<Arc<[usize]>>,
        dtype: Dtype,
    ) -> Result<Self, SaveError> {
        if dtype == Dtype::Unknown {
            return Err(SaveError::InvalidInput(
                "cannot use Unknown dtype for tensor".to_owned(),
            ));
        }

        let shape = shape.into();
        let stride = stride.into();

        if shape.len() != stride.len() {
            return Err(SaveError::InvalidInput(format!(
                "shape length ({}) must match stride length ({})",
                shape.len(),
                stride.len()
            )));
        }

        let _ = offset
            .checked_add(size)
            .ok_or_else(|| SaveError::InvalidInput("offset + size overflow".to_owned()))?;

        Ok(Self {
            offset,
            size,
            shape,
            stride,
            dtype,
        })
    }

    /// Byte offset within the storage region.
    #[inline]
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Size in bytes.
    #[inline]
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Shape in elements per dimension.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Stride in elements per dimension.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// Element type.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_common_dtypes() {
        assert_eq!(Dtype::from_str("F32").unwrap(), Dtype::F32);
        assert_eq!(Dtype::from_str("u8").unwrap(), Dtype::U8);
        assert_eq!(Dtype::from_str("BF16").unwrap(), Dtype::Bf16);
    }

    #[test]
    fn display_returns_lower() {
        assert_eq!(Dtype::F32.to_string(), "f32");
        assert_eq!(Dtype::Bf16.to_string(), "bf16");
    }

    #[test]
    fn unknown_dtype_round_trips() {
        assert_eq!(Dtype::Unknown.as_str(), "unknown");
    }

    #[test]
    fn tensor_meta_rejects_unknown_dtype() {
        let err = TensorMeta::new(0, 4, vec![2usize, 2], vec![2usize, 1], Dtype::Unknown);
        assert!(err.is_err());
    }

    #[test]
    fn tensor_meta_rejects_shape_stride_mismatch() {
        let err = TensorMeta::new(0, 4, vec![2usize, 2], vec![1usize], Dtype::F32);
        assert!(err.is_err());
    }

    #[test]
    fn tensor_meta_rejects_overflow() {
        let err = TensorMeta::new(u64::MAX, 1, vec![1usize], vec![1usize], Dtype::F32);
        assert!(err.is_err());
    }

    #[test]
    fn tensor_meta_accessors() {
        let meta = TensorMeta::new(8, 16, vec![4usize], vec![1usize], Dtype::F64).unwrap();
        assert_eq!(meta.offset(), 8);
        assert_eq!(meta.size(), 16);
        assert_eq!(meta.shape(), &[4]);
        assert_eq!(meta.stride(), &[1]);
        assert_eq!(meta.dtype(), Dtype::F64);
    }

    #[test]
    fn tensor_meta_scalar_zero_dims() {
        // Scalar tensor: shape=[], stride=[] is valid
        let meta =
            TensorMeta::new(0, 4, Vec::<usize>::new(), Vec::<usize>::new(), Dtype::F32).unwrap();
        assert_eq!(meta.shape(), &[] as &[usize]);
        assert_eq!(meta.stride(), &[] as &[usize]);
    }
}
