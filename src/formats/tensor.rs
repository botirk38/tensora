//! Shared tensor primitives across all formats.

use std::fmt;
use std::str::FromStr;

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
}
