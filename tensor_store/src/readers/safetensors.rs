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
//! use tensor_store::readers::safetensors;
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
//! // Load with mmap (Linux only, lazy)
//! let tensors_mmap = safetensors::load_mmap("model.safetensors").await?;
//! let tensors = tensors_mmap.tensors(); // Access parsed structure
//! ```

use crate::backends;
use crate::readers::error::{ReaderError, ReaderResult};
use crate::readers::traits::{AsyncReader, SyncReader, TensorMetadata};
pub use safetensors::SafeTensorError;
pub use safetensors::tensor::{Dtype, SafeTensors, TensorView as Tensor};
use std::ops::Deref;
use std::path::Path;

/// SafeTensors reader with mmap-backed storage (lazy loading).
///
/// This reader memory-maps the file and parses the SafeTensors header lazily.
/// Tensor data is accessed directly from the memory map without copying.
/// Available on all platforms that support memory mapping.
#[non_exhaustive]
pub struct SafeTensorsMmap {
    mmap: backends::mmap::Mmap,
    tensors: SafeTensors<'static>,
}

/// Owned SafeTensors reader with buffer-backed storage.
///
/// This reader loads the entire file into memory and owns the data.
/// Provides fast access to all tensors with eager parsing.
#[non_exhaustive]
pub struct SafeTensorsOwned {
    buffer: Box<[u8]>,
    tensors: SafeTensors<'static>,
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

        // SAFETY: The slice points into `buffer`, which we store inside the struct.
        // Drop order is `tensors` first, then `buffer`, so the data lives for the
        // entire lifetime of the SafeTensors object.
        let static_slice: &'static [u8] = unsafe { std::mem::transmute(slice) };
        let tensors = SafeTensors::deserialize(static_slice)?;

        Ok(Self { buffer, tensors })
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

        // SAFETY: The slice points into the mmap, which we store inside the struct.
        // Drop order is `tensors` first, then `mmap`, so the data lives for the
        // entire lifetime of the SafeTensorsMmap object.
        let static_slice: &'static [u8] = unsafe { std::mem::transmute(slice) };
        let tensors = SafeTensors::deserialize(static_slice)?;

        Ok(Self { mmap, tensors })
    }

    /// Access the parsed `SafeTensors` structure.
    #[inline]
    #[must_use]
    pub const fn tensors(&self) -> &SafeTensors<'static> {
        &self.tensors
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
        self.tensors.names().into_iter().any(|n| n == name)
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.names()
    }
}

impl AsyncReader for SafeTensorsMmap {
    type Output = Self;

    #[inline]
    async fn load(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned())
        })?;
        let mmap = backends::mmap::map(path_str)?;
        Self::from_mmap(mmap)
    }
}

impl SyncReader for SafeTensorsMmap {
    type Output = Self;

    #[inline]
    fn load_sync(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
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
        self.tensors.names().into_iter().any(|n| n == name)
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.names()
    }
}

impl AsyncReader for SafeTensorsOwned {
    type Output = Self;

    #[inline]
    async fn load(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned())
        })?;
        let bytes = backends::load(path_str).await?;
        Self::from_bytes(bytes)
    }
}

impl SyncReader for SafeTensorsOwned {
    type Output = Self;

    #[inline]
    fn load_sync(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned())
        })?;
        let bytes = backends::sync::load(path_str)?;
        Self::from_bytes(bytes)
    }
}

/// Load tensor data using the best backend for the current platform and parse it.
///
/// Returns a `SafeTensorsOwned` with the parsed tensors.
#[inline]
pub async fn load(path: impl AsRef<Path>) -> ReaderResult<SafeTensorsOwned> {
    SafeTensorsOwned::load(path).await
}

/// Load tensor data in parallel chunks and parse it.
///
/// Returns a `SafeTensorsOwned` with the parsed tensors.
#[inline]
pub async fn load_parallel(
    path: impl AsRef<Path>,
    chunks: usize,
) -> ReaderResult<SafeTensorsOwned> {
    let path_str = path
        .as_ref()
        .to_str()
        .ok_or_else(|| ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned()))?;
    let bytes = backends::load_parallel(path_str, chunks).await?;
    SafeTensorsOwned::from_bytes(bytes)
}

/// Synchronous load using mmap (Linux) or `std::fs` (other platforms).
///
/// Returns a `SafeTensorsOwned` with the parsed tensors.
#[inline]
pub fn load_sync(path: impl AsRef<Path>) -> ReaderResult<SafeTensorsOwned> {
    SafeTensorsOwned::load_sync(path)
}

/// Synchronous ranged load using mmap (Linux) or `std::fs` (other platforms).
///
/// Returns a `SafeTensorsOwned` with the parsed tensors.
#[inline]
pub fn load_range_sync(
    path: impl AsRef<Path>,
    offset: u64,
    len: usize,
) -> ReaderResult<SafeTensorsOwned> {
    let path_str = path
        .as_ref()
        .to_str()
        .ok_or_else(|| ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned()))?;
    let bytes = backends::sync::load_range(path_str, offset, len)?;
    SafeTensorsOwned::from_bytes(bytes)
}

/// Load tensor data using memory mapping (lazy loading).
///
/// Returns a `SafeTensorsMmap` with memory-mapped tensors.
#[inline]
pub async fn load_mmap(path: impl AsRef<Path>) -> ReaderResult<SafeTensorsMmap> {
    SafeTensorsMmap::load(path).await
}

/// Load tensor data synchronously using memory mapping (lazy loading).
///
/// Returns a `SafeTensorsMmap` with memory-mapped tensors.
#[inline]
pub fn load_mmap_sync(path: impl AsRef<Path>) -> ReaderResult<SafeTensorsMmap> {
    SafeTensorsMmap::load_sync(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn test_load_parallel_zero_copy() {
        // Test with the existing test file
        let path = "test_model.safetensors";

        tokio_uring::start(async {
            // Load with parallel loading (zero-copy)
            let data = load_parallel(path, 4).await.unwrap();

            // Verify we can deserialize it
            let tensors = data.tensors();
            assert!(!tensors.names().is_empty());

            // Load with single-threaded loading for comparison
            let data_single = load(path).await.unwrap();

            // Data should be identical
            assert_eq!(data.as_bytes().len(), data_single.as_bytes().len());
            assert_eq!(data.as_bytes(), data_single.as_bytes());

            println!(
                "Zero-copy parallel loading test passed! Loaded {} tensors",
                tensors.names().len()
            );
        });
    }

    #[cfg(not(target_os = "linux"))]
    #[tokio::test]
    async fn test_load_parallel_zero_copy() {
        // Test with the existing test file
        let path = "test_model.safetensors";

        // Load with parallel loading (zero-copy)
        let data = load_parallel(path, 4).await.unwrap();

        // Verify we can deserialize it
        let tensors = data.tensors();
        assert!(!tensors.names().is_empty());

        // Load with single-threaded loading for comparison
        let data_single = load(path).await.unwrap();

        // Data should be identical
        assert_eq!(data.as_bytes().len(), data_single.as_bytes().len());
        assert_eq!(data.as_bytes(), data_single.as_bytes());

        println!(
            "Zero-copy parallel loading test passed! Loaded {} tensors",
            tensors.names().len()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_load_parallel_zero_copy_io_uring() {
        // Test io_uring version specifically
        let path = "test_model.safetensors";

        tokio_uring::start(async {
            // Load with parallel loading (zero-copy)
            let data = load_parallel(path, 4).await.unwrap();

            // Verify we can deserialize it
            let tensors = data.tensors();
            assert!(!tensors.names().is_empty());

            // Load with single-threaded loading for comparison
            let data_single = load(path).await.unwrap();

            // Data should be identical
            assert_eq!(data.as_bytes().len(), data_single.as_bytes().len());
            assert_eq!(data.as_bytes(), data_single.as_bytes());

            println!(
                "Zero-copy parallel loading (io_uring) test passed! Loaded {} tensors",
                tensors.names().len()
            );
        });
    }
}
