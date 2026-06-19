//! ServerlessLLM identity and count types.

use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};

/// Identifies a single ServerlessLLM partition file (`tensor.data_<N>`).
///
/// Partitions are numbered from zero. `PartitionId(0)` corresponds to
/// `tensor.data_0`, `PartitionId(1)` to `tensor.data_1`, and so on.
///
/// # Composability
///
/// `PartitionId` is a plain value object — copy, comparable, hashable. Use it
/// as a map key, array index (via `.as_usize()`), or display value. The only
/// format-specific helper is [`data_file_stem`][PartitionId::data_file_stem],
/// which encodes the `tensor.data_<N>` naming convention for this format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartitionId(usize);

impl PartitionId {
    /// Wraps a raw partition index.
    #[inline]
    #[must_use]
    pub const fn new(n: usize) -> Self {
        Self(n)
    }

    /// Returns the underlying index.
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0
    }

    /// Returns the data file stem for this partition: `"tensor.data_<N>"`.
    ///
    /// Used by both the writer (checkpoint) and the reader (model/index) to
    /// build partition file paths without duplicating the naming convention.
    #[must_use]
    pub fn data_file_stem(self) -> String {
        format!("tensor.data_{}", self.0)
    }

    /// Returns the next partition ID (`self + 1`).
    ///
    /// Useful when iterating over a known count: `(0..count).map(PartitionId::new)`.
    #[inline]
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for PartitionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for PartitionId {
    #[inline]
    fn from(n: usize) -> Self {
        Self(n)
    }
}

impl From<PartitionId> for usize {
    #[inline]
    fn from(id: PartitionId) -> Self {
        id.0
    }
}

// ---------------------------------------------------------------------------

/// Total number of ServerlessLLM partitions, always at least one.
///
/// Constructed via [`PartitionCount::new`] (returns `None` for zero) or
/// [`PartitionCount::one`]. Converts freely to/from `usize` and
/// `NonZeroUsize`.
///
/// # Composability
///
/// `PartitionCount` carries the invariant `count >= 1` so callers never need
/// to check for the degenerate zero-partition case. Use
/// [`iter`][PartitionCount::iter] to iterate over all valid `PartitionId`s
/// for this count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartitionCount(NonZeroUsize);

impl PartitionCount {
    /// Returns a count of one partition.
    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        // SAFETY: 1 != 0
        Self(unsafe { NonZeroUsize::new_unchecked(1) })
    }

    /// Returns `Some(PartitionCount(n))` if `n >= 1`, otherwise `None`.
    #[inline]
    #[must_use]
    pub const fn new(n: usize) -> Option<Self> {
        match NonZeroUsize::new(n) {
            Some(nz) => Some(Self(nz)),
            None => None,
        }
    }

    /// Returns the count as a `usize`.
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0.get()
    }

    /// Returns an iterator over all `PartitionId`s `0..self`.
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = PartitionId> {
        (0..self.0.get()).map(PartitionId::new)
    }
}

impl fmt::Display for PartitionCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<NonZeroUsize> for PartitionCount {
    #[inline]
    fn from(n: NonZeroUsize) -> Self {
        Self(n)
    }
}

impl From<PartitionCount> for usize {
    #[inline]
    fn from(c: PartitionCount) -> Self {
        c.0.get()
    }
}

impl From<PartitionCount> for NonZeroUsize {
    #[inline]
    fn from(c: PartitionCount) -> Self {
        c.0
    }
}

// ---------------------------------------------------------------------------
// PartitionSizing
// ---------------------------------------------------------------------------

/// Layout sizing policy for ServerlessLLM partitions.
///
/// This value object allows configuring the target bytes per partition for
/// automatic layout decisions.
#[derive(Debug, Clone, Copy)]
pub struct PartitionSizing {
    target_bytes: NonZeroU64,
}

impl PartitionSizing {
    /// Default target bytes per partition (512 MiB).
    pub const DEFAULT_TARGET_BYTES: u64 = 512 * 1024 * 1024;

    /// Returns a sizing policy with the default target (512 MiB).
    #[must_use]
    pub fn default_target() -> Self {
        Self {
            target_bytes: NonZeroU64::new(Self::DEFAULT_TARGET_BYTES).expect("512 MiB is non-zero"),
        }
    }

    /// Create a sizing policy with a custom target bytes.
    ///
    /// # Errors
    ///
    /// Returns `None` if `target_bytes` is zero.
    pub fn with_target_bytes(target_bytes: u64) -> Option<Self> {
        Some(Self {
            target_bytes: NonZeroU64::new(target_bytes)?,
        })
    }

    /// Returns the target bytes per partition for this sizing policy.
    #[inline]
    #[must_use]
    pub fn target_bytes(&self) -> NonZeroU64 {
        self.target_bytes
    }

    /// Returns the target bytes as a `u64`.
    #[inline]
    #[must_use]
    pub fn target_bytes_u64(&self) -> u64 {
        self.target_bytes.get()
    }

    /// Recommended partition count for a model of `total_bytes`.
    ///
    /// Formula: `max(1, ceil(total_bytes / target_bytes))` with no artificial upper
    /// bound (beyond `usize`).
    #[must_use]
    pub fn recommended_count(&self, total_bytes: u64) -> PartitionCount {
        if total_bytes == 0 {
            return PartitionCount::one();
        }
        let n = total_bytes.div_ceil(self.target_bytes.get());
        let n_usize = if n > usize::MAX as u64 {
            usize::MAX
        } else {
            n as usize
        };
        PartitionCount::new(n_usize).unwrap_or_else(PartitionCount::one)
    }
}

impl Default for PartitionSizing {
    fn default() -> Self {
        Self::default_target()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_id_round_trips() {
        let id = PartitionId::new(3);
        assert_eq!(id.as_usize(), 3);
        assert_eq!(usize::from(id), 3);
        assert_eq!(PartitionId::from(3usize), id);
    }

    #[test]
    fn partition_id_data_file_stem() {
        assert_eq!(PartitionId::new(0).data_file_stem(), "tensor.data_0");
        assert_eq!(PartitionId::new(7).data_file_stem(), "tensor.data_7");
    }

    #[test]
    fn partition_id_next() {
        assert_eq!(PartitionId::new(2).next(), PartitionId::new(3));
    }

    #[test]
    fn partition_id_display() {
        assert_eq!(PartitionId::new(5).to_string(), "5");
    }

    #[test]
    fn partition_id_ordering() {
        assert!(PartitionId::new(0) < PartitionId::new(1));
        assert_eq!(PartitionId::new(2), PartitionId::new(2));
    }

    #[test]
    fn partition_count_new_zero_is_none() {
        assert!(PartitionCount::new(0).is_none());
    }

    #[test]
    fn partition_count_new_nonzero() {
        let c = PartitionCount::new(4).unwrap();
        assert_eq!(c.as_usize(), 4);
        assert_eq!(usize::from(c), 4);
    }

    #[test]
    fn partition_count_one() {
        assert_eq!(PartitionCount::one().as_usize(), 1);
    }

    #[test]
    fn partition_count_iter() {
        let ids: Vec<PartitionId> = PartitionCount::new(3).unwrap().iter().collect();
        assert_eq!(
            ids,
            vec![
                PartitionId::new(0),
                PartitionId::new(1),
                PartitionId::new(2)
            ]
        );
    }

    #[test]
    fn partition_count_display() {
        assert_eq!(PartitionCount::new(8).unwrap().to_string(), "8");
    }

    // PartitionSizing tests
    #[test]
    fn default_target_bytes() {
        let sizing = PartitionSizing::default_target();
        assert_eq!(sizing.target_bytes_u64(), 512 * 1024 * 1024);
    }

    #[test]
    fn custom_target_bytes() {
        let sizing = PartitionSizing::with_target_bytes(1024 * 1024).unwrap();
        assert_eq!(sizing.target_bytes_u64(), 1024 * 1024);
    }

    #[test]
    fn zero_target_bytes_is_none() {
        assert!(PartitionSizing::with_target_bytes(0).is_none());
    }

    #[test]
    fn default_is_default_target() {
        let sizing = PartitionSizing::default();
        assert_eq!(
            sizing.target_bytes_u64(),
            PartitionSizing::DEFAULT_TARGET_BYTES
        );
    }

    #[test]
    fn recommended_partition_count_zero_and_small() {
        let sizing = PartitionSizing::default_target();
        assert_eq!(sizing.recommended_count(0).as_usize(), 1);
        assert_eq!(sizing.recommended_count(1).as_usize(), 1);
        assert_eq!(
            sizing
                .recommended_count(PartitionSizing::DEFAULT_TARGET_BYTES - 1)
                .as_usize(),
            1
        );
        assert_eq!(
            sizing
                .recommended_count(PartitionSizing::DEFAULT_TARGET_BYTES)
                .as_usize(),
            1
        );
    }

    #[test]
    fn recommended_partition_count_scales_by_target() {
        let sizing = PartitionSizing::default_target();
        assert_eq!(
            sizing
                .recommended_count(PartitionSizing::DEFAULT_TARGET_BYTES + 1)
                .as_usize(),
            2
        );
        assert_eq!(
            sizing
                .recommended_count(2 * PartitionSizing::DEFAULT_TARGET_BYTES)
                .as_usize(),
            2
        );
        assert_eq!(
            sizing
                .recommended_count(2 * PartitionSizing::DEFAULT_TARGET_BYTES + 1)
                .as_usize(),
            3
        );
    }

    #[test]
    fn recommended_partition_count_large() {
        let sizing = PartitionSizing::default_target();
        let total = 32 * 1024 * 1024 * 1024u64;
        let count = sizing.recommended_count(total);
        assert_eq!(count.as_usize(), 64);
    }

    #[test]
    fn custom_target_affects_count() {
        let sizing = PartitionSizing::with_target_bytes(256 * 1024 * 1024).unwrap(); // 256 MiB
        // With smaller target, same total needs more partitions
        let total = 1024 * 1024 * 1024u64; // 1 GiB
        assert_eq!(sizing.recommended_count(total).as_usize(), 4);

        let larger = PartitionSizing::with_target_bytes(1024 * 1024 * 1024).unwrap(); // 1 GiB
        assert_eq!(larger.recommended_count(total).as_usize(), 1);
    }
}
