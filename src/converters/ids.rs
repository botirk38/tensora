//! Converter statistics count types.

use std::fmt;

/// Number of tensors involved in a conversion.
///
/// Appears in [`ConversionStats`][crate::converters::safetensors_to_serverlessllm::ConversionStats]
/// as a named type so the count cannot be silently confused with a byte count,
/// a partition count, or a shard count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorCount(usize);

impl TensorCount {
    /// Wraps a raw count.
    #[inline]
    #[must_use]
    pub const fn new(n: usize) -> Self {
        Self(n)
    }

    /// Returns the count as `usize`.
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0
    }

    /// Returns `true` when the count is zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for TensorCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for TensorCount {
    #[inline]
    fn from(n: usize) -> Self {
        Self(n)
    }
}

impl From<TensorCount> for usize {
    #[inline]
    fn from(c: TensorCount) -> Self {
        c.0
    }
}

// ---------------------------------------------------------------------------

/// Number of individual copy operations in a [`ConversionPlan`].
///
/// Each copy op moves one tensor's bytes from a source shard range to a
/// destination partition offset. This count is distinct from `TensorCount`
/// because future layouts may split a single tensor across multiple ops.
///
/// [`ConversionPlan`]: crate::converters::safetensors_to_serverlessllm::ConversionPlan
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CopyOpCount(usize);

impl CopyOpCount {
    /// Wraps a raw count.
    #[inline]
    #[must_use]
    pub const fn new(n: usize) -> Self {
        Self(n)
    }

    /// Returns the count as `usize`.
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0
    }

    /// Returns `true` when there are no copy operations.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for CopyOpCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for CopyOpCount {
    #[inline]
    fn from(n: usize) -> Self {
        Self(n)
    }
}

impl From<CopyOpCount> for usize {
    #[inline]
    fn from(c: CopyOpCount) -> Self {
        c.0
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_count_round_trips() {
        let c = TensorCount::new(42);
        assert_eq!(c.as_usize(), 42);
        assert_eq!(usize::from(c), 42);
        assert_eq!(TensorCount::from(42usize), c);
    }

    #[test]
    fn tensor_count_zero() {
        assert!(TensorCount::new(0).is_zero());
        assert!(!TensorCount::new(1).is_zero());
    }

    #[test]
    fn tensor_count_display() {
        assert_eq!(TensorCount::new(7).to_string(), "7");
    }

    #[test]
    fn copy_op_count_round_trips() {
        let c = CopyOpCount::new(10);
        assert_eq!(c.as_usize(), 10);
        assert_eq!(usize::from(c), 10);
        assert_eq!(CopyOpCount::from(10usize), c);
    }

    #[test]
    fn copy_op_count_zero() {
        assert!(CopyOpCount::new(0).is_zero());
        assert!(!CopyOpCount::new(1).is_zero());
    }

    #[test]
    fn copy_op_count_display() {
        assert_eq!(CopyOpCount::new(3).to_string(), "3");
    }
}
