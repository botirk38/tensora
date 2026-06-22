//! ServerlessLLM identity and count types.

use std::fmt;
use std::num::NonZeroUsize;

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
}
