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
pub use safetensors::SafeTensorError;
pub use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::ops::Deref;

/// SafeTensors data plus the owned backing buffer.
///
/// The buffer is kept alive for as long as the parsed [`SafeTensors`] lives,
/// ensuring the borrowed tensor views remain valid.
pub struct OwnedSafeTensors {
    buffer: Box<[u8]>,
    tensors: SafeTensors<'static>,
}

impl OwnedSafeTensors {
    fn from_bytes(bytes: Vec<u8>) -> Result<Self, SafeTensorError> {
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
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Consume the owned tensors and return the serialized bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer.into()
    }

    /// Access the parsed SafeTensors structure.
    pub fn tensors(&self) -> &SafeTensors<'static> {
        &self.tensors
    }
}

impl Deref for OwnedSafeTensors {
    type Target = SafeTensors<'static>;

    fn deref(&self) -> &Self::Target {
        &self.tensors
    }
}

#[cfg(target_os = "linux")]
async fn load_bytes(path: &str) -> crate::IoResult<Vec<u8>> {
    backends::io_uring::load(path).await
}

#[cfg(not(target_os = "linux"))]
async fn load_bytes(path: &str) -> crate::IoResult<Vec<u8>> {
    backends::async_io::load(path).await
}

#[cfg(target_os = "linux")]
async fn load_bytes_parallel(path: &str, chunks: usize) -> crate::IoResult<Vec<u8>> {
    backends::io_uring::load_parallel(path, chunks).await
}

#[cfg(not(target_os = "linux"))]
async fn load_bytes_parallel(path: &str, chunks: usize) -> crate::IoResult<Vec<u8>> {
    backends::async_io::load_parallel(path, chunks).await
}

/// Load tensor data using the best backend for the current platform and parse it.
#[inline]
pub async fn load(path: &str) -> Result<OwnedSafeTensors, SafeTensorError> {
    let bytes = load_bytes(path).await.map_err(SafeTensorError::IoError)?;
    OwnedSafeTensors::from_bytes(bytes)
}

/// Load tensor data in parallel chunks and parse it.
#[inline]
pub async fn load_parallel(path: &str, chunks: usize) -> Result<OwnedSafeTensors, SafeTensorError> {
    let bytes = load_bytes_parallel(path, chunks)
        .await
        .map_err(SafeTensorError::IoError)?;
    OwnedSafeTensors::from_bytes(bytes)
}
