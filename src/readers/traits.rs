//! Traits for tensor readers.

use crate::readers::error::ReaderResult;
use std::path::Path;

/// Trait for asynchronous tensor readers.
#[allow(async_fn_in_trait)]
pub trait AsyncReader {
    /// The output type produced by this reader.
    type Output;

    /// Asynchronously loads tensor data from the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    async fn load(path: impl AsRef<Path>) -> ReaderResult<Self::Output>;
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
    fn load_sync(path: impl AsRef<Path>) -> ReaderResult<Self::Output>;
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

#[cfg(test)]
mod tests {
    use super::*;

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
