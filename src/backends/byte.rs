//! Owned byte buffer types for backend I/O.
//!
//! This module provides `OwnedBytes`, a zero-copy owned byte container
//! that can be backed by different storage types (pooled, aligned, shared, mmap).

use std::sync::Arc;

pub use zeropool::PooledBuffer;

use super::mmap::Mmap;
#[cfg(target_os = "linux")]
use super::odirect::AlignedBuffer;

/// Owned byte buffer that can be backed by different storage types.
/// Supports zero-copy access patterns while preserving ownership semantics.
pub enum OwnedBytes {
    Pooled(PooledBuffer),
    #[cfg(target_os = "linux")]
    Aligned(AlignedBuffer),
    Shared(Arc<[u8]>),
    Mmap(Mmap),
    /// Plain Vec-backed buffer for backends that need mutable storage.
    Vec(Vec<u8>),
}

impl OwnedBytes {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Pooled(b) => b.len(),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => b.len(),
            Self::Shared(b) => b.len(),
            Self::Mmap(b) => b.len(),
            Self::Vec(b) => b.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Raw mutable pointer for kernel I/O submission.
    /// Safe to use only for Pooled, Aligned, and Vec variants.
    /// Panics on Shared and Mmap.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Pooled(b) => b.as_mut_ptr(),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => b.as_mut_ptr(),
            Self::Shared(_) => panic!("OwnedBytes::Shared is not mutable"),
            Self::Mmap(_) => panic!("OwnedBytes::Mmap is not mutable"),
            Self::Vec(b) => b.as_mut_ptr(),
        }
    }

    /// Mutable slice for safe backend access.
    /// Returns None for Shared and Mmap variants.
    #[inline]
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::Pooled(b) => Some(b.as_mut_slice()),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => Some(b.as_mut_slice()),
            Self::Shared(_) => None,
            Self::Mmap(_) => None,
            Self::Vec(b) => Some(b.as_mut_slice()),
        }
    }

    /// Convert to `Vec<u8>`. Copies for aligned and mmap-backed storage
    /// to avoid UB from mismatched allocator layouts.
    pub fn into_vec(self) -> Vec<u8> {
        match self {
            Self::Pooled(b) => b.into_inner(),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => b.as_slice().to_vec(),
            Self::Shared(b) => b.to_vec(),
            Self::Mmap(b) => b.as_slice().to_vec(),
            Self::Vec(b) => b,
        }
    }

    /// Convert to `Arc<[u8]>`. Copies for aligned and mmap-backed storage
    /// to avoid UB from mismatched allocator layouts.
    pub fn into_shared(self) -> Arc<[u8]> {
        match self {
            Self::Pooled(b) => b.into_inner().into(),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => b.as_slice().into(),
            Self::Shared(b) => b,
            Self::Mmap(b) => Arc::from(b.as_slice()),
            Self::Vec(b) => b.into(),
        }
    }

    /// Create an OwnedBytes from a plain Vec<u8>.
    #[inline]
    pub fn from_vec(v: Vec<u8>) -> Self {
        Self::Vec(v)
    }
}

impl AsRef<[u8]> for OwnedBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Pooled(b) => b.as_ref(),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => b.as_slice(),
            Self::Shared(b) => b.as_ref(),
            Self::Mmap(b) => b.as_slice(),
            Self::Vec(b) => b.as_ref(),
        }
    }
}

impl std::ops::Deref for OwnedBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl std::fmt::Debug for OwnedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedBytes")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}
