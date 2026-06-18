//! SafeTensors shard identity and count types.

use std::fmt;
use std::num::NonZeroUsize;

/// Identifies a single SafeTensors shard file within a multi-shard model.
///
/// Shards are numbered from zero in the order they are discovered on disk
/// (sorted by file name). `ShardId(0)` is the first shard alphabetically,
/// `ShardId(1)` the second, and so on.
///
/// # Composability
///
/// `ShardId` is a plain value object — copy, comparable, hashable. Use it as a
/// `Vec` index (via [`.as_usize()`][ShardId::as_usize]) or map key. The
/// newtype prevents accidentally treating a shard index as a partition index or
/// any other domain count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardId(usize);

impl ShardId {
    /// Wraps a raw shard index.
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
}

impl fmt::Display for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for ShardId {
    #[inline]
    fn from(n: usize) -> Self {
        Self(n)
    }
}

impl From<ShardId> for usize {
    #[inline]
    fn from(id: ShardId) -> Self {
        id.0
    }
}

// ---------------------------------------------------------------------------

/// Total number of shards in a multi-file SafeTensors model, always at least one.
///
/// Constructed via [`ShardCount::new`] (returns `None` for zero) or
/// [`ShardCount::one`]. A single-file model has `ShardCount::one()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardCount(NonZeroUsize);

impl ShardCount {
    /// Returns a count of one shard (single-file model).
    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        // SAFETY: 1 != 0
        Self(unsafe { NonZeroUsize::new_unchecked(1) })
    }

    /// Returns `Some(ShardCount(n))` if `n >= 1`, otherwise `None`.
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

    /// Returns an iterator over all `ShardId`s `0..self`.
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = ShardId> {
        (0..self.0.get()).map(ShardId::new)
    }

    /// Returns `true` if this is a single-shard model.
    #[inline]
    #[must_use]
    pub fn is_single(self) -> bool {
        self.0.get() == 1
    }
}

impl fmt::Display for ShardCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<NonZeroUsize> for ShardCount {
    #[inline]
    fn from(n: NonZeroUsize) -> Self {
        Self(n)
    }
}

impl From<ShardCount> for usize {
    #[inline]
    fn from(c: ShardCount) -> Self {
        c.0.get()
    }
}

impl From<ShardCount> for NonZeroUsize {
    #[inline]
    fn from(c: ShardCount) -> Self {
        c.0
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_id_round_trips() {
        let id = ShardId::new(2);
        assert_eq!(id.as_usize(), 2);
        assert_eq!(usize::from(id), 2);
        assert_eq!(ShardId::from(2usize), id);
    }

    #[test]
    fn shard_id_display() {
        assert_eq!(ShardId::new(4).to_string(), "4");
    }

    #[test]
    fn shard_id_ordering() {
        assert!(ShardId::new(0) < ShardId::new(1));
        assert_eq!(ShardId::new(3), ShardId::new(3));
    }

    #[test]
    fn shard_count_new_zero_is_none() {
        assert!(ShardCount::new(0).is_none());
    }

    #[test]
    fn shard_count_new_nonzero() {
        let c = ShardCount::new(3).unwrap();
        assert_eq!(c.as_usize(), 3);
        assert_eq!(usize::from(c), 3);
    }

    #[test]
    fn shard_count_one() {
        assert_eq!(ShardCount::one().as_usize(), 1);
        assert!(ShardCount::one().is_single());
    }

    #[test]
    fn shard_count_iter() {
        let ids: Vec<ShardId> = ShardCount::new(3).unwrap().iter().collect();
        assert_eq!(ids, vec![ShardId::new(0), ShardId::new(1), ShardId::new(2)]);
    }

    #[test]
    fn shard_count_display() {
        assert_eq!(ShardCount::new(5).unwrap().to_string(), "5");
    }
}
