//! SafeTensors format reader.
//!
//! This module re-exports types from the safetensors crate for convenience.
//! All parsing is handled by the safetensors library.
//!
//! # Usage
//!
//! ```rust,ignore
//! use tensor_store::readers::safetensors;
//!
//! // Load and parse SafeTensors file
//! let tensors = safetensors::load("model.safetensors").await?;
//!
//! // Access tensors
//! for name in tensors.names() {
//!     let view = tensors.tensor(name)?;
//!     println!("{}: {:?} ({})", name, view.shape(), view.dtype());
//! }
//! ```

use crate::backends;
use crate::readers::error::{ReaderError, ReaderResult};
use crate::readers::traits::{AsyncReader, SyncReader, TensorMetadata};
pub use safetensors::SafeTensorError;
pub use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::ops::Deref;
use std::path::Path;

/// SafeTensors data plus the owned backing buffer.
///
/// The buffer is kept alive for as long as the parsed [`SafeTensors`] lives,
/// ensuring the borrowed tensor views remain valid.
#[non_exhaustive]
pub struct OwnedSafeTensors {
    buffer: Box<[u8]>,
    tensors: SafeTensors<'static>,
}

impl OwnedSafeTensors {
    /// Creates an owned SafeTensors from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes cannot be parsed as SafeTensors format.
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
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Consume the owned tensors and return the serialized bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer.into()
    }

    /// Access the parsed SafeTensors structure.
    #[inline]
    pub fn tensors(&self) -> &SafeTensors<'static> {
        &self.tensors
    }
}

impl Clone for OwnedSafeTensors {
    fn clone(&self) -> Self {
        // We can safely clone by copying the buffer and re-parsing
        Self::from_bytes(self.buffer.to_vec())
            .expect("cloning already-parsed SafeTensors should not fail")
    }
}

impl Deref for OwnedSafeTensors {
    type Target = SafeTensors<'static>;

    fn deref(&self) -> &Self::Target {
        &self.tensors
    }
}

impl AsRef<SafeTensors<'static>> for OwnedSafeTensors {
    fn as_ref(&self) -> &SafeTensors<'static> {
        &self.tensors
    }
}

impl TryFrom<Vec<u8>> for OwnedSafeTensors {
    type Error = ReaderError;

    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        Self::from_bytes(bytes)
    }
}

impl TensorMetadata for OwnedSafeTensors {
    fn len(&self) -> usize {
        self.tensors.len()
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.names().into_iter().any(|n| n == name)
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.names()
    }
}

impl AsyncReader for OwnedSafeTensors {
    type Output = Self;

    async fn load(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let bytes = backends::load(path.as_ref().to_str().unwrap()).await?;
        Self::from_bytes(bytes)
    }
}

impl SyncReader for OwnedSafeTensors {
    type Output = Self;

    fn load_sync(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let bytes = backends::sync::load(path.as_ref().to_str().unwrap())?;
        Self::from_bytes(bytes)
    }
}

/// Load tensor data using the best backend for the current platform and parse it.
#[inline]
pub async fn load(path: impl AsRef<Path>) -> ReaderResult<OwnedSafeTensors> {
    OwnedSafeTensors::load(path).await
}

/// Load tensor data in parallel chunks and parse it.
#[inline]
pub async fn load_parallel(
    path: impl AsRef<Path>,
    chunks: usize,
) -> ReaderResult<OwnedSafeTensors> {
    let bytes = backends::load_parallel(path.as_ref().to_str().unwrap(), chunks).await?;
    OwnedSafeTensors::from_bytes(bytes)
}

/// Synchronous load using mmap (Linux) or std::fs (other platforms).
#[inline]
pub fn load_sync(path: impl AsRef<Path>) -> ReaderResult<OwnedSafeTensors> {
    OwnedSafeTensors::load_sync(path)
}

/// Synchronous ranged load using mmap (Linux) or std::fs (other platforms).
#[inline]
pub fn load_range_sync(
    path: impl AsRef<Path>,
    offset: u64,
    len: usize,
) -> ReaderResult<OwnedSafeTensors> {
    let bytes = backends::sync::load_range(path.as_ref().to_str().unwrap(), offset, len)?;
    OwnedSafeTensors::from_bytes(bytes)
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
