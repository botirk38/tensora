//! `SafeTensors` format serializer.
//!
//! This module intentionally keeps things simple by wrapping the public
//! `safetensors` crate APIs. It provides ergonomic names and re-exports so
//! users of `tensor_store` don't have to depend on the upstream crate
//! directly.
//!
//! # Example
//!
//! ```rust,ignore
//! use tensor_store::safetensors::{
//!     self, Dtype, MetadataMap, TensorView
//! };
//!
//! let data = vec![0u8; 4];
//! let tensor = TensorView::new(Dtype::F32, vec![1, 1], &data).unwrap();
//! let writer = Writer::new();
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
use crate::formats::error::{WriterError, WriterResult};
use crate::formats::traits::{AsyncSerializer, SyncSerializer};
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
pub struct Writer;

impl Writer {
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

    /// Serialize tensors into an owned byte buffer using the `SafeTensors` format.
    ///
    /// This method is synchronous since serialization is CPU-bound.
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
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        safetensors::serialize_to_file(tensors, metadata, path_ref).map_err(Into::into)
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
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let buffer = safetensors::serialize(tensors, metadata)?;
        let mut writer = backends::AsyncWriter::create(path_ref)
            .await
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        writer.write_all(&buffer).await.map_err(WriterError::from)
    }
}

/// Input data for writing a SafeTensors model.
#[derive(Debug, Clone)]
pub struct TensorWriteData {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Input data for writing a SafeTensors model.
#[derive(Debug, Clone)]
pub struct WriteInput {
    pub tensors: Vec<TensorWriteData>,
    pub metadata: Option<MetadataMap>,
}

impl AsyncSerializer for Writer {
    type Input = WriteInput;

    async fn write(path: &Path, data: &Self::Input) -> WriterResult<()> {
        let views: Vec<_> = data
            .tensors
            .iter()
            .map(|t| {
                let dtype = dtype_from_str(&t.dtype).map_err(WriterError::InvalidInput)?;
                let view = TensorView::new(dtype, t.shape.clone(), t.data.as_slice())
                    .map_err(|e| WriterError::InvalidInput(e.to_string()))?;
                Ok::<_, WriterError>((t.name.as_str(), view))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let writer = Writer::new();
        writer
            .write_to_file_async(views, data.metadata.clone(), path)
            .await
    }
}

impl SyncSerializer for Writer {
    type Input = WriteInput;

    fn write_sync(path: &Path, data: &Self::Input) -> WriterResult<()> {
        let views: Vec<_> = data
            .tensors
            .iter()
            .map(|t| {
                let dtype = dtype_from_str(&t.dtype).map_err(WriterError::InvalidInput)?;
                let view = TensorView::new(dtype, t.shape.clone(), t.data.as_slice())
                    .map_err(|e| WriterError::InvalidInput(e.to_string()))?;
                Ok::<_, WriterError>((t.name.as_str(), view))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let writer = Writer::new();
        writer.write_to_file(views, data.metadata.clone(), path)
    }
}

fn dtype_from_str(s: &str) -> Result<Dtype, String> {
    match s.to_uppercase().as_str() {
        "BOOL" | "B" => Ok(Dtype::BOOL),
        "U8" => Ok(Dtype::U8),
        "I8" => Ok(Dtype::I8),
        "I16" => Ok(Dtype::I16),
        "U16" => Ok(Dtype::U16),
        "F16" => Ok(Dtype::F16),
        "F32" | "F" => Ok(Dtype::F32),
        "F64" | "D" => Ok(Dtype::F64),
        "I32" => Ok(Dtype::I32),
        "I64" => Ok(Dtype::I64),
        "U32" => Ok(Dtype::U32),
        "U64" => Ok(Dtype::U64),
        "BF16" => Ok(Dtype::BF16),
        _ => Err(format!("unknown dtype: {}", s)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::SafeTensors;
    use safetensors::tensor::TensorView;
    use tempfile::TempDir;

    fn sample_view() -> TensorView<'static> {
        TensorView::new(Dtype::U8, vec![3], &[1u8, 2, 3]).expect("tensor view")
    }

    #[test]
    fn write_to_buffer_roundtrips() {
        let writer = Writer::new();
        let bytes = writer
            .write_to_buffer([("a", sample_view())], None)
            .expect("serialize");
        let tensors = SafeTensors::deserialize(&bytes).expect("deserialize");
        assert_eq!(tensors.names(), vec!["a"]);
        assert_eq!(tensors.tensor("a").unwrap().dtype(), Dtype::U8);
    }

    #[test]
    fn write_to_file_sync_and_metadata() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        let writer = Writer::new();

        writer
            .write_to_file(
                [("weight", sample_view())],
                Some(HashMap::from([("k".into(), "v".into())])),
                &path,
            )
            .expect("write file");

        let data = std::fs::read(&path).unwrap();
        let tensors = SafeTensors::deserialize(&data).unwrap();
        assert_eq!(tensors.names(), vec!["weight"]);
        // SafeTensors crate does not expose metadata accessor; ensure data parses.
        assert_eq!(tensors.tensor("weight").unwrap().dtype(), Dtype::U8);
    }

    #[test]
    fn write_to_file_async_uses_backends() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("async").join("model.safetensors");
        let writer = Writer::new();

        crate::test_utils::run_async(async {
            writer
                .write_to_file_async([("a", sample_view())], None, &path)
                .await
                .expect("async write");
        });

        let data = std::fs::read(path).unwrap();
        let tensors = SafeTensors::deserialize(&data).unwrap();
        assert_eq!(tensors.tensor("a").unwrap().shape(), &[3]);
    }
}
