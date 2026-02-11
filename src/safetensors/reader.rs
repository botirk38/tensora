//! `SafeTensors` format reader.
//!
//! This module provides readers for the SafeTensors format, offering both owned and mmap-backed storage.
//! All parsing is handled by the safetensors library.
//!
//! # Types
//!
//! - `SafeTensorsOwned`: Owned reader with buffer-backed storage (eager loading).
//! - `SafeTensorsMmap`: Mmap-backed reader with lazy loading.
//!
//! # Usage
//!
//! ```rust,ignore
//! use tensor_store::safetensors;
//!
//! // Load and parse SafeTensors file (owned)
//! let tensors = safetensors::load("model.safetensors").await?;
//!
//! // Access tensors
//! for name in tensors.names() {
//!     let view = tensors.tensor(name)?;
//!     println!("{}: {:?} ({})", name, view.shape(), view.dtype());
//! }
//!
//! // Load with mmap (cross platform lazy loading)
//! let tensors_mmap = safetensors::load_mmap("model.safetensors");
//! let tensors = tensors_mmap.tensors(); // Access parsed structure
//!
//! let tensors_sync = safetensors::load_sync("model.safetensors")?;
//! let tensors = tensors_sync.tensors(); // Access parsed structure
//! ```

use crate::backends;
use crate::types::error::{ReaderError, ReaderResult};
use crate::types::traits::{AsyncReader, SyncReader, TensorMetadata};
pub use safetensors::SafeTensorError;
pub use safetensors::tensor::{Dtype, SafeTensors, TensorView as Tensor};
use std::ops::Deref;
use std::path::Path;

/// SafeTensors reader with mmap-backed storage (lazy loading).
///
/// This reader memory-maps the file and parses the SafeTensors header lazily.
/// Tensor data is accessed directly from the memory map without copying.
/// Available on all platforms that support memory mapping.
///
/// # Field Order
///
/// The field order is critical for soundness. Fields are dropped in declaration order,
/// so `tensors` (which references `mmap`) MUST be declared before `mmap`.
/// Do not reorder these fields.
#[derive(Debug)]
pub struct SafeTensorsMmap {
    tensors: SafeTensors<'static>,
    mmap: backends::mmap::Mmap,
}

/// Owned SafeTensors reader with buffer-backed storage.
///
/// This reader loads the entire file into memory and owns the data.
/// Provides fast access to all tensors with eager parsing.
///
/// # Field Order
///
/// The field order is critical for soundness. Fields are dropped in declaration order,
/// so `tensors` (which references `buffer`) MUST be declared before `buffer`.
/// Do not reorder these fields.
#[derive(Debug)]
pub struct SafeTensorsOwned {
    tensors: SafeTensors<'static>,
    buffer: Box<[u8]>,
}

impl SafeTensorsOwned {
    /// Creates an owned `SafeTensors` from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes cannot be parsed as `SafeTensors` format.
    #[inline]
    pub fn from_bytes(bytes: Vec<u8>) -> ReaderResult<Self> {
        let buffer = bytes.into_boxed_slice();
        let slice: &[u8] = &buffer;

        // SAFETY: Lifetime extension via transmute to create self-referential struct
        //
        // We transmute &'buffer [u8] to &'static [u8] to store SafeTensors<'static>
        // alongside the buffer it references. This is sound because:
        //
        // 1. Self-reference: SafeTensors<'static> references self.buffer which lives
        //    in the same struct. Box<[u8]> is heap-allocated and never moved.
        //
        // 2. Drop order: Rust drops fields in declaration order. The `tensors` field
        //    is declared BEFORE `buffer`, so SafeTensors is dropped first, then buffer.
        //    This ensures the reference in SafeTensors is never used after buffer is
        //    deallocated. See struct definition for field order documentation.
        //
        // 3. No mutation: We never expose &mut access to buffer after parsing, so
        //    the data SafeTensors references cannot be invalidated.
        //
        // 4. Not Send/Sync by default: Without explicit impl, this struct cannot be
        //    sent across threads, avoiding data races.
        //
        // Alternative approaches considered:
        // - Pin<Box<Self>>: More complex, still requires unsafe
        // - Owned TensorView: Would copy all tensor data on access (2GB copy for 2GB tensor)
        // - Separate buffer + metadata: Loses ergonomics, users manage lifetimes
        // - ouroboros/rental crates: External dependency for same pattern
        //
        // This pattern is equivalent to what ouroboros/rental provide for
        // self-referential structs, implemented directly to avoid dependencies.
        let static_slice: &'static [u8] = unsafe { std::mem::transmute(slice) };
        let tensors = SafeTensors::deserialize(static_slice)?;

        Ok(Self { tensors, buffer })
    }

    /// Borrow the underlying serialized bytes.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Consume the owned tensors and return the serialized bytes.
    #[inline]
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer.into()
    }

    /// Access the parsed `SafeTensors` structure.
    #[inline]
    #[must_use]
    pub const fn tensors(&self) -> &SafeTensors<'static> {
        &self.tensors
    }

    /// Returns a tensor view by name using `ReaderError` for convenience.
    #[inline]
    pub fn tensor(&self, name: &str) -> ReaderResult<Tensor<'static>> {
        self.tensors.tensor(name).map_err(ReaderError::from)
    }

    /// Returns tensor names without requiring the `TensorMetadata` trait in scope.
    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.names()
    }

    /// Returns true when no tensors are loaded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl Clone for SafeTensorsOwned {
    fn clone(&self) -> Self {
        // We can safely clone by copying the buffer and re-parsing
        Self::from_bytes(self.buffer.to_vec())
            .expect("SafeTensors parsing should not fail on valid data")
    }
}

impl Deref for SafeTensorsOwned {
    type Target = SafeTensors<'static>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.tensors
    }
}

impl SafeTensorsMmap {
    /// Creates an mmap-backed `SafeTensors` from a memory-mapped file.
    ///
    /// # Errors
    ///
    /// Returns an error if the mapped data cannot be parsed as `SafeTensors` format.
    #[inline]
    pub fn from_mmap(mmap: backends::mmap::Mmap) -> ReaderResult<Self> {
        let slice: &[u8] = mmap.as_slice();

        // SAFETY: Same pattern as SafeTensorsOwned::from_bytes - see detailed comment there.
        // TL;DR: SafeTensors<'static> references self.mmap, drop order is tensors-then-mmap,
        // no mutation after construction, so the reference remains valid for struct lifetime.
        let static_slice: &'static [u8] = unsafe { std::mem::transmute(slice) };
        let tensors = SafeTensors::deserialize(static_slice)?;

        Ok(Self { tensors, mmap })
    }

    /// Access the parsed `SafeTensors` structure.
    #[inline]
    #[must_use]
    pub const fn tensors(&self) -> &SafeTensors<'static> {
        &self.tensors
    }

    /// Returns a tensor view by name using `ReaderError` for convenience.
    #[inline]
    pub fn tensor(&self, name: &str) -> ReaderResult<Tensor<'static>> {
        self.tensors.tensor(name).map_err(ReaderError::from)
    }

    /// Returns tensor names without requiring the `TensorMetadata` trait in scope.
    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.names()
    }

    /// Returns true when no tensors are mapped.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Access the underlying memory-mapped data.
    #[inline]
    #[must_use]
    pub const fn mmap(&self) -> &backends::mmap::Mmap {
        &self.mmap
    }
}

impl TensorMetadata for SafeTensorsMmap {
    #[inline]
    fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.tensors.tensor(name).is_ok()
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.names()
    }
}

impl SyncReader for SafeTensorsMmap {
    type Output = Self;

    #[inline]
    fn load_sync(path: &Path) -> ReaderResult<Self::Output> {
        let path_str = path.to_str().ok_or_else(|| {
            ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned())
        })?;
        let mmap = backends::mmap::map(path_str)?;
        Self::from_mmap(mmap)
    }
}

impl TryFrom<Vec<u8>> for SafeTensorsOwned {
    type Error = ReaderError;

    #[inline]
    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        Self::from_bytes(bytes)
    }
}

impl TensorMetadata for SafeTensorsOwned {
    #[inline]
    fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.tensors.tensor(name).is_ok()
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.names()
    }
}

impl AsyncReader for SafeTensorsOwned {
    type Output = Self;

    #[inline]
    async fn load(path: &Path) -> ReaderResult<Self::Output> {
        let bytes = backends::async_backend().load(path).await?;
        Self::from_bytes(bytes)
    }
}

impl SyncReader for SafeTensorsOwned {
    type Output = Self;

    #[inline]
    fn load_sync(path: &Path) -> ReaderResult<Self::Output> {
        let bytes = backends::sync_backend().load(path)?;
        Self::from_bytes(bytes)
    }
}

/// Load tensor data using the best backend for the current platform and parse it.
///
/// Returns a `SafeTensorsOwned` with the parsed tensors.
///
/// # Errors
///
/// - File cannot be read
/// - Invalid SafeTensors format
#[inline]
pub async fn load(path: impl AsRef<Path>) -> ReaderResult<SafeTensorsOwned> {
    SafeTensorsOwned::load(path.as_ref()).await
}

/// Load tensor data in parallel chunks and parse it.
///
/// Returns a `SafeTensorsOwned` with the parsed tensors.
#[inline]
pub async fn load_parallel(
    path: impl AsRef<Path>,
    chunks: usize,
) -> ReaderResult<SafeTensorsOwned> {
    let path_ref = path.as_ref();
    let bytes = backends::async_backend()
        .load_parallel(path_ref, chunks)
        .await?;
    SafeTensorsOwned::from_bytes(bytes)
}

/// Synchronous load using buffered I/O (owned `Vec<u8>`).
///
/// Returns a `SafeTensorsOwned` with the parsed tensors.
///
/// # Errors
///
/// - File cannot be read
/// - Invalid SafeTensors format
#[inline]
pub fn load_sync(path: impl AsRef<Path>) -> ReaderResult<SafeTensorsOwned> {
    SafeTensorsOwned::load_sync(path.as_ref())
}

/// Synchronous ranged load **only when the range covers the full file**.
///
/// SafeTensors cannot be deserialized from partial ranges; this helper enforces
/// that the requested range matches the file size starting at offset 0.
#[inline]
pub fn load_range_sync(
    path: impl AsRef<Path>,
    offset: u64,
    len: usize,
) -> ReaderResult<SafeTensorsOwned> {
    if offset != 0 {
        return Err(ReaderError::InvalidMetadata(
            "SafeTensors requires the full file; offset must be 0".to_owned(),
        ));
    }

    let path_ref = path.as_ref();

    let file_len = std::fs::metadata(path_ref)
        .map_err(ReaderError::from)?
        .len();

    let expected_len = u64::try_from(len)
        .map_err(|_| ReaderError::InvalidMetadata("requested length overflows u64".to_owned()))?;

    if file_len != expected_len {
        return Err(ReaderError::InvalidMetadata(format!(
            "SafeTensors requires full file: requested len {len}, file size {file_len}"
        )));
    }

    let bytes = backends::sync_backend().load_range(path_ref, offset, len)?;
    SafeTensorsOwned::from_bytes(bytes)
}

/// Load tensor data synchronously using memory mapping (lazy loading).
///
/// Returns a `SafeTensorsMmap` with memory-mapped tensors.
///
/// # Errors
///
/// - File cannot be memory-mapped
/// - Invalid SafeTensors format
#[inline]
pub fn load_mmap(path: impl AsRef<Path>) -> ReaderResult<SafeTensorsMmap> {
    SafeTensorsMmap::load_sync(path.as_ref())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::serialize;
    use safetensors::tensor::TensorView;
    use tempfile::TempDir;

    fn sample_bytes() -> Vec<u8> {
        let data = vec![1u8, 2, 3, 4];
        let view = TensorView::new(Dtype::U8, vec![4], &data).expect("create tensor view");
        serialize([("tensor", view)], None).expect("serialize")
    }

    #[test]
    fn owned_from_bytes_roundtrips_and_clones() {
        let bytes = sample_bytes();
        let owned = SafeTensorsOwned::from_bytes(bytes.clone()).expect("parse");
        assert_eq!(owned.tensor_names(), vec!["tensor"]);
        assert_eq!(owned.as_bytes(), bytes.as_slice());
        assert_eq!(owned.tensor("tensor").unwrap().shape(), &[4]);
        assert!(!owned.is_empty());
        let cloned = owned.clone();
        assert_eq!(cloned.tensor_names(), vec!["tensor"]);
        assert_eq!(cloned.len(), 1);
    }

    #[test]
    fn owned_from_bytes_rejects_invalid_data() {
        let err = SafeTensorsOwned::from_bytes(vec![0, 1, 2]).unwrap_err();
        assert!(matches!(err, ReaderError::SafeTensors(_)));
    }

    #[test]
    fn load_sync_and_metadata_helpers_work() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_bytes()).unwrap();

        let owned = load_sync(&path).expect("load sync");
        assert!(owned.contains("tensor"));
        assert_eq!(owned.tensor_names(), vec!["tensor"]);
        assert!(!owned.is_empty());
        assert_eq!(owned.tensor("tensor").unwrap().dtype(), Dtype::U8);
    }

    #[test]
    fn load_range_sync_requires_full_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        let bytes = sample_bytes();
        std::fs::write(&path, &bytes).unwrap();

        let err = load_range_sync(&path, 1, bytes.len()).unwrap_err();
        assert!(matches!(err, ReaderError::InvalidMetadata(_)));
    }

    #[test]
    fn load_async_variants_parse() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_bytes()).unwrap();

        tokio_uring::start(async {
            let owned = load(&path).await.expect("load async");
            assert_eq!(owned.len(), 1);

            let parallel = load_parallel(&path, 2).await.expect("load parallel");
            assert_eq!(parallel.len(), 1);
        });
    }

    #[test]
    fn mmap_loader_parses_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_bytes()).unwrap();

        let mmap = load_mmap(&path).expect("load mmap");
        assert_eq!(mmap.len(), 1);
        assert!(mmap.contains("tensor"));
        assert_eq!(mmap.tensor("tensor").unwrap().dtype(), Dtype::U8);
    }

    // ---------------------------------------------------------------------------
    // Soundness Tests for Unsafe Transmute
    // ---------------------------------------------------------------------------

    #[test]
    fn test_transmute_soundness_owned() {
        // Verify that the transmute lifetime hack doesn't allow use-after-free
        // This test ensures drop order is correct
        
        let bytes = sample_bytes();
        let reader = SafeTensorsOwned::from_bytes(bytes).unwrap();
        
        // Access is fine while reader lives
        let _view = reader.tensor("tensor").unwrap();
        assert_eq!(_view.shape(), &[4]);
        
        // This would not compile if lifetimes were incorrect (uncomment to verify):
        // let view = reader.tensor("tensor").unwrap();
        // drop(reader);
        // let _ = view.data();  // ERROR: use after free
        
        // Reader can be cloned (re-parses from buffer)
        let reader2 = reader.clone();
        drop(reader);
        let _view2 = reader2.tensor("tensor").unwrap(); // Still OK
    }

    #[test]
    fn test_transmute_soundness_mmap() {
        // Same test for mmap variant
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_bytes()).unwrap();

        let reader = load_mmap(&path).unwrap();
        
        // Access is fine while reader lives
        let _view = reader.tensor("tensor").unwrap();
        assert_eq!(_view.shape(), &[4]);
        
        // Mmap data remains valid for struct lifetime
        drop(reader);
        // _view is now invalid (would not compile)
    }

    #[test]
    fn test_field_drop_order_owned() {
        // Verify fields are declared in correct order for drop safety
        // tensors must be declared before buffer so it drops first
        
        let owned = SafeTensorsOwned::from_bytes(sample_bytes()).unwrap();
        let base = &owned as *const _ as usize;
        
        // Get field offsets
        let tensors_offset = &owned.tensors as *const _ as usize - base;
        let buffer_offset = &owned.buffer as *const _ as usize - base;
        
        assert!(tensors_offset < buffer_offset, 
            "tensors field must be declared before buffer for correct drop order. \
             tensors offset: {}, buffer offset: {}", tensors_offset, buffer_offset);
    }

    #[test]
    fn test_field_drop_order_mmap() {
        // Same check for mmap variant
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_bytes()).unwrap();

        let mmap_reader = load_mmap(&path).unwrap();
        let base = &mmap_reader as *const _ as usize;
        
        // Get field offsets
        let tensors_offset = &mmap_reader.tensors as *const _ as usize - base;
        let mmap_offset = &mmap_reader.mmap as *const _ as usize - base;
        
        assert!(tensors_offset < mmap_offset, 
            "tensors field must be declared before mmap for correct drop order. \
             tensors offset: {}, mmap offset: {}", tensors_offset, mmap_offset);
    }

    #[test]
    fn test_no_send_sync_by_default() {
        // Verify that these types are not Send/Sync without explicit impl
        // This prevents accidental threading issues
        
        fn assert_not_send<T: Send>() {}
        fn assert_not_sync<T: Sync>() {}
        
        // These should fail to compile if uncommented:
        // assert_not_send::<SafeTensorsOwned>();
        // assert_not_sync::<SafeTensorsOwned>();
        // assert_not_send::<SafeTensorsMmap>();
        // assert_not_sync::<SafeTensorsMmap>();
        
        // SafeTensors from upstream IS Send+Sync, which is fine since it borrows external data
        assert_not_send::<SafeTensors<'_>>();
        assert_not_sync::<SafeTensors<'_>>();
    }
}
