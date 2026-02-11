//! Traits for tensor readers and writers.

use crate::types::error::{ReaderResult, WriterResult};
use std::path::Path;

// ============================================================================
// Reader Traits
// ============================================================================

/// Trait for asynchronous tensor readers.
///
/// Note: The returned futures are not required to implement `Send` because some
/// backends (like `tokio-uring`) use thread-local runtimes.
#[allow(async_fn_in_trait)]
pub trait AsyncReader {
    /// The output type produced by this reader.
    type Output;

    /// Asynchronously loads tensor data from the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    async fn load(path: &Path) -> ReaderResult<Self::Output>;
}

/// Trait for synchronous tensor readers.
pub trait SyncReader {
    /// The output type produced by this reader.
    type Output;

    /// Synchronously loads tensor data from the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    fn load_sync(path: &Path) -> ReaderResult<Self::Output>;
}

/// Trait for types that provide tensor metadata.
pub trait TensorMetadata {
    /// Returns the number of tensors.
    fn len(&self) -> usize;

    /// Returns true if there are no tensors.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if a tensor with the given name exists.
    fn contains(&self, name: &str) -> bool;

    /// Returns the names of all tensors.
    fn tensor_names(&self) -> Vec<&str>;
}

// ============================================================================
// Writer Traits
// ============================================================================

/// Trait for asynchronous tensor writers.
///
/// Note: The returned futures are not required to implement `Send` because some
/// backends use thread-local runtimes.
#[allow(async_fn_in_trait)]
pub trait AsyncWriter {
    /// The input data type accepted by this writer.
    type Input;

    /// Asynchronously writes tensor data to the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - The output path where the tensor data will be written
    /// * `data` - The tensor data to write
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or if the data is invalid.
    async fn write(path: &Path, data: &Self::Input) -> WriterResult<()>;
}

/// Trait for synchronous tensor writers.
pub trait SyncWriter {
    /// The input data type accepted by this writer.
    type Input;

    /// Synchronously writes tensor data to the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - The output path where the tensor data will be written
    /// * `data` - The tensor data to write
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or if the data is invalid.
    fn write_sync(path: &Path, data: &Self::Input) -> WriterResult<()>;
}

#[cfg(test)]
mod tests {
    use super::TensorMetadata;

    struct DummyMeta(Vec<String>);

    impl TensorMetadata for DummyMeta {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn contains(&self, name: &str) -> bool {
            self.0.iter().any(|n| n == name)
        }

        fn tensor_names(&self) -> Vec<&str> {
            self.0.iter().map(String::as_str).collect()
        }
    }

    #[test]
    fn tensor_metadata_defaults_work() {
        let meta = DummyMeta(vec!["a".into(), "b".into()]);
        assert!(!meta.is_empty());
        assert!(meta.contains("a"));
        assert_eq!(meta.tensor_names(), vec!["a", "b"]);

        let empty = DummyMeta(vec![]);
        assert!(empty.is_empty());
    }
}
