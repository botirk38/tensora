//! Traits for tensor readers and writers.

use crate::formats::error::WriterResult;
use std::path::Path;



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

    /// Returns tensor names as a borrowed slice (cached, sorted).
    fn tensor_names(&self) -> &[std::sync::Arc<str>];
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


// ============================================================================
// Tensor View Trait
// ============================================================================

/// Uniform read-only view of a single tensor's data.
///
/// Both format families implement this so callers can work with tensors
/// without knowing which format produced them.
pub trait TensorView {
    /// Shape in elements.
    fn shape(&self) -> &[usize];
    /// Dtype as a canonical string (e.g. `"float32"`, `"int64"`).
    fn dtype(&self) -> &str;
    /// Raw bytes of the tensor data.
    fn data(&self) -> &[u8];
}

#[cfg(test)]
mod tests {
    use super::TensorMetadata;
    use std::sync::Arc;

    struct DummyMeta(Vec<Arc<str>>);

    impl DummyMeta {
        fn new(names: Vec<&str>) -> Self {
            Self(names.into_iter().map(|s| s.into()).collect())
        }
    }

    impl TensorMetadata for DummyMeta {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn contains(&self, name: &str) -> bool {
            self.0.iter().any(|n| n.as_ref() == name)
        }

        fn tensor_names(&self) -> &[Arc<str>] {
            &self.0
        }
    }

    #[test]
    fn tensor_metadata_defaults_work() {
        let meta = DummyMeta::new(vec!["a", "b"]);
        assert!(!meta.is_empty());
        assert!(meta.contains("a"));
        assert_eq!(meta.tensor_names().len(), 2);
        assert_eq!(meta.tensor_names()[0].as_ref(), "a");
        assert_eq!(meta.tensor_names()[1].as_ref(), "b");

        let empty = DummyMeta::new(vec![]);
        assert!(empty.is_empty());
    }
}
