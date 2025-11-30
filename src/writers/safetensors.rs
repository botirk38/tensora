//! `SafeTensors` writer helpers.
//!
//! This module intentionally keeps things simple by wrapping the public
//! `safetensors` crate APIs. It provides ergonomic names and re-exports so
//! users of `tensor_store` don't have to depend on the upstream crate
//! directly.
//!
//! # Example
//!
//! ```rust,ignore
//! use tensor_store::writers::safetensors::{
//!     self, Dtype, MetadataMap, TensorView
//! };
//!
//! let data = vec![0u8; 4];
//! let tensor = TensorView::new(Dtype::F32, vec![1, 1], &data).unwrap();
//! let writer = SafeTensorsWriter::new();
//!
//! // Sync usage
//! let bytes = writer.write_to_buffer([("weight", tensor)], None).unwrap();
//! writer.write_to_file([("weight", tensor)], None, "model.safetensors").unwrap();
//!
//! // Async usage
//! let bytes = writer.write_to_buffer_async([("weight", tensor)], None).await.unwrap();
//! writer.write_to_file_async([("weight", tensor)], None, "model.safetensors").await.unwrap();
//! ```

use crate::backends;
use crate::writers::error::WriterResult;
use std::collections::HashMap;
use std::fmt::Display;
use std::path::Path;

pub use safetensors::tensor::{TensorView, View};
pub use safetensors::{Dtype, SafeTensorError};

/// Convenience alias for custom metadata passed to `SafeTensors`.
pub type MetadataMap = HashMap<String, String>;

/// Stateless helper that proxies calls to the upstream `SafeTensors` serializer.
///
/// This writer is intentionally stateless and can be copied freely.
/// It provides both synchronous and asynchronous methods for writing `SafeTensors` format.
#[derive(Debug, Default, Clone, Copy)]
pub struct SafeTensorsWriter;

impl SafeTensorsWriter {
    /// Create a new writer instance.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Serialize tensors into an owned byte buffer using the `SafeTensors` format (synchronous).
    ///
    /// This is a thin wrapper around [`safetensors::serialize`].
    ///
    /// # Arguments
    ///
    /// * `tensors` - Iterator of (name, `tensor_view`) pairs to serialize
    /// * `metadata` - Optional custom metadata to include in the file
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[inline]
    pub fn write_to_buffer<S, V, I>(
        &self,
        tensors: I,
        metadata: Option<MetadataMap>,
    ) -> WriterResult<Vec<u8>>
    where
        S: AsRef<str> + Ord + Display,
        V: View,
        I: IntoIterator<Item = (S, V)>,
    {
        safetensors::serialize(tensors, metadata).map_err(Into::into)
    }

    /// Serialize tensors into an owned byte buffer using the `SafeTensors` format (asynchronous).
    ///
    /// This is identical to the sync version since serialization is CPU-bound.
    /// Use this for consistency in async contexts.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Iterator of (name, `tensor_view`) pairs to serialize
    /// * `metadata` - Optional custom metadata to include in the file
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[inline]
    pub fn write_to_buffer_sync<S, V, I>(
        &self,
        tensors: I,
        metadata: Option<MetadataMap>,
    ) -> WriterResult<Vec<u8>>
    where
        S: AsRef<str> + Ord + Display,
        V: View,
        I: IntoIterator<Item = (S, V)>,
    {
        safetensors::serialize(tensors, metadata).map_err(Into::into)
    }

    /// Serialize tensors directly to a `.safetensors` file on disk (synchronous).
    ///
    /// This delegates to [`safetensors::serialize_to_file`] so no extra buffering
    /// happens inside `tensor_store`.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Iterator of (name, `tensor_view`) pairs to serialize
    /// * `metadata` - Optional custom metadata to include in the file
    /// * `path` - Path where the file will be written
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    #[inline]
    pub fn write_to_file<S, V, I, P>(
        &self,
        tensors: I,
        metadata: Option<MetadataMap>,
        path: P,
    ) -> WriterResult<()>
    where
        S: AsRef<str> + Ord + Display,
        V: View,
        I: IntoIterator<Item = (S, V)>,
        P: AsRef<Path>,
    {
        safetensors::serialize_to_file(tensors, metadata, path.as_ref()).map_err(Into::into)
    }

    /// Serialize tensors directly to a `.safetensors` file on disk (asynchronous).
    ///
    /// This serializes to a buffer first, then uses the optimized backends module
    /// for async file writing.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Iterator of (name, `tensor_view`) pairs to serialize
    /// * `metadata` - Optional custom metadata to include in the file
    /// * `path` - Path where the file will be written
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub async fn write_to_file_async<S, V, I, P>(
        &self,
        tensors: I,
        metadata: Option<MetadataMap>,
        path: P,
    ) -> WriterResult<()>
    where
        S: AsRef<str> + Ord + Display,
        V: View,
        I: IntoIterator<Item = (S, V)>,
        P: AsRef<Path>,
    {
        let buffer = safetensors::serialize(tensors, metadata)?;
        backends::write_all(path.as_ref(), buffer)
            .await
            .map_err(Into::into)
    }
}
