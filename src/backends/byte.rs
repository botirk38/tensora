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
    pub fn from_pooled(buf: PooledBuffer) -> Self {
        Self::Pooled(buf)
    }

    #[cfg(target_os = "linux")]
    #[inline]
    pub fn from_aligned(buf: AlignedBuffer) -> Self {
        Self::Aligned(buf)
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_mmap() -> Mmap {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"mmap_data").unwrap();
        tmp.flush().unwrap();
        super::super::mmap::map(tmp.path()).unwrap()
    }

    // --- construction ---

    #[test]
    fn from_vec_roundtrips() {
        let data = vec![10u8, 20, 30];
        let ob = OwnedBytes::from_vec(data.clone());
        assert_eq!(ob.as_ref(), &data[..]);
    }

    #[test]
    fn from_pooled_roundtrips() {
        let pool = super::super::get_buffer_pool();
        let mut buf = pool.get(8);
        buf[..4].copy_from_slice(&[1, 2, 3, 4]);
        let ob = OwnedBytes::from_pooled(buf);
        assert_eq!(&ob.as_ref()[..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn shared_variant_from_arc() {
        let data: Arc<[u8]> = Arc::from(vec![5u8, 6, 7]);
        let ob = OwnedBytes::Shared(data.clone());
        assert_eq!(ob.as_ref(), data.as_ref());
    }

    #[test]
    fn mmap_variant_accessible() {
        let ob = OwnedBytes::Mmap(make_mmap());
        assert_eq!(ob.as_ref(), b"mmap_data");
    }

    // --- accessors ---

    #[test]
    fn len_and_is_empty_vec() {
        let ob = OwnedBytes::from_vec(vec![1, 2]);
        assert_eq!(ob.len(), 2);
        assert!(!ob.is_empty());

        let empty = OwnedBytes::from_vec(vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn len_and_is_empty_shared() {
        let ob = OwnedBytes::Shared(Arc::new([]));
        assert!(ob.is_empty());
        assert_eq!(ob.len(), 0);
    }

    #[test]
    fn as_ref_and_deref_match() {
        let data = vec![42u8; 16];
        let ob = OwnedBytes::from_vec(data.clone());
        let via_as_ref: &[u8] = ob.as_ref();
        let via_deref: &[u8] = &ob;
        assert_eq!(via_as_ref, via_deref);
        assert_eq!(via_as_ref, &data[..]);
    }

    #[test]
    fn debug_shows_len() {
        let ob = OwnedBytes::from_vec(vec![0; 42]);
        let dbg = format!("{:?}", ob);
        assert!(dbg.contains("42"), "debug output should include length: {dbg}");
    }

    // --- mutability ---

    #[test]
    fn as_mut_slice_some_for_vec() {
        let mut ob = OwnedBytes::from_vec(vec![0; 4]);
        let slice = ob.as_mut_slice().expect("Vec should be mutable");
        slice[0] = 99;
        assert_eq!(ob.as_ref()[0], 99);
    }

    #[test]
    fn as_mut_slice_none_for_shared() {
        let mut ob = OwnedBytes::Shared(Arc::from(vec![1u8, 2]));
        assert!(ob.as_mut_slice().is_none());
    }

    #[test]
    fn as_mut_slice_none_for_mmap() {
        let mut ob = OwnedBytes::Mmap(make_mmap());
        assert!(ob.as_mut_slice().is_none());
    }

    #[test]
    fn as_mut_ptr_works_for_vec() {
        let mut ob = OwnedBytes::from_vec(vec![0; 4]);
        let ptr = ob.as_mut_ptr();
        assert!(!ptr.is_null());
    }

    #[test]
    #[should_panic(expected = "Shared is not mutable")]
    fn as_mut_ptr_panics_on_shared() {
        let mut ob = OwnedBytes::Shared(Arc::from(vec![1u8]));
        let _ = ob.as_mut_ptr();
    }

    #[test]
    #[should_panic(expected = "Mmap is not mutable")]
    fn as_mut_ptr_panics_on_mmap() {
        let mut ob = OwnedBytes::Mmap(make_mmap());
        let _ = ob.as_mut_ptr();
    }

    // --- conversions ---

    #[test]
    fn into_vec_preserves_bytes_vec() {
        let data = vec![1u8, 2, 3];
        let ob = OwnedBytes::from_vec(data.clone());
        assert_eq!(ob.into_vec(), data);
    }

    #[test]
    fn into_vec_preserves_bytes_shared() {
        let data: Arc<[u8]> = Arc::from(vec![4u8, 5, 6]);
        let ob = OwnedBytes::Shared(data.clone());
        assert_eq!(ob.into_vec(), data.to_vec());
    }

    #[test]
    fn into_vec_preserves_bytes_mmap() {
        let ob = OwnedBytes::Mmap(make_mmap());
        assert_eq!(ob.into_vec(), b"mmap_data");
    }

    #[test]
    fn into_shared_preserves_bytes_vec() {
        let data = vec![7u8, 8, 9];
        let ob = OwnedBytes::from_vec(data.clone());
        assert_eq!(ob.into_shared().as_ref(), &data[..]);
    }

    #[test]
    fn into_shared_preserves_bytes_shared() {
        let data: Arc<[u8]> = Arc::from(vec![10u8, 11]);
        let ob = OwnedBytes::Shared(data.clone());
        let shared = ob.into_shared();
        assert_eq!(shared.as_ref(), data.as_ref());
    }

    #[test]
    fn into_shared_preserves_bytes_pooled() {
        let pool = super::super::get_buffer_pool();
        let mut buf = pool.get(4);
        buf[..4].copy_from_slice(&[1, 2, 3, 4]);
        let ob = OwnedBytes::from_pooled(buf);
        let shared = ob.into_shared();
        assert_eq!(&shared[..4], &[1, 2, 3, 4]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn aligned_buffer_roundtrips() {
        let mut aligned = super::super::odirect::AlignedBuffer::new(4096).unwrap();
        aligned.set_len(4);
        aligned.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 4]);
        let ob = OwnedBytes::from_aligned(aligned);
        assert_eq!(ob.len(), 4);
        assert_eq!(&ob.as_ref()[..4], &[1, 2, 3, 4]);
        assert_eq!(ob.into_vec(), vec![1, 2, 3, 4]);
    }
}
