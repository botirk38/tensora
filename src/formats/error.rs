//! Error types for tensor readers and writers.

use std::io;

// ============================================================================
// Reader Errors
// ============================================================================

/// Unified error type for all tensor readers.
#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    /// I/O error occurred during reading.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// `SafeTensors` format error.
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    /// Tensor not found in the index.
    #[error("Tensor '{name}' not found in index")]
    TensorNotFound {
        /// Name of the tensor that was not found.
        name: String,
    },

    /// Partition file not found or inaccessible.
    #[error("Partition {partition_id} not found at path '{path}'")]
    PartitionNotFound {
        /// ID of the partition.
        partition_id: usize,
        /// Expected path to the partition file.
        path: String,
    },

    /// Partition file is smaller than required for the requested tensor.
    #[error("Partition file '{path}' is too small: has {actual} bytes, needs {required} bytes")]
    PartitionTooSmall {
        /// Path to the partition file.
        path: String,
        /// Actual size of the partition file.
        actual: u64,
        /// Required size for the tensor.
        required: u64,
    },

    /// Invalid tensor index format (JSON structure error).
    #[error("Invalid tensor index format: {0}")]
    InvalidIndexFormat(String),

    /// JSON parse error when reading tensor index.
    #[error("JSON parse error in tensor index: {0}")]
    JsonParseError(String),

    /// Offset + size calculation would overflow.
    #[error("Offset + size overflow for tensor '{name}'")]
    OffsetOverflow {
        /// Name of the tensor.
        name: String,
    },

    /// Tensor size exceeds platform limits.
    #[error("Size too large for tensor '{name}': {size} bytes")]
    SizeTooLarge {
        /// Name of the tensor.
        name: String,
        /// Size that exceeded limits.
        size: u64,
    },

    /// Mutex was poisoned (another thread panicked while holding the lock).
    #[error("Mutex lock poisoned")]
    MutexPoisoned,

    /// Generic `ServerlessLLM` format error (for cases not covered by specific variants).
    #[error("ServerlessLLM format error: {0}")]
    ServerlessLlm(String),

    /// `TensorStore` format error.
    #[error("TensorStore format error: {0}")]
    TensorStore(String),

    /// Invalid tensor metadata.
    #[error("Invalid tensor metadata: {0}")]
    InvalidMetadata(String),
}

/// Result type alias for reader operations.
pub type ReaderResult<T> = Result<T, ReaderError>;

// ============================================================================
// Writer Errors
// ============================================================================

/// Errors that can occur during tensor writing operations.
#[derive(Debug, thiserror::Error)]
pub enum WriterError {
    /// I/O error occurred during write operations.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Error from the `SafeTensors` library.
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    /// Error specific to `ServerlessLLM` format writing.
    #[error("ServerlessLLM error: {0}")]
    ServerlessLlm(String),

    /// Error specific to `TensorStore` format writing.
    #[error("TensorStore error: {0}")]
    TensorStore(String),

    /// Invalid input provided to writer.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Serialization error occurred.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Path-related error.
    #[error("Path error: {0}")]
    Path(String),
}

/// A specialized Result type for writer operations.
pub type WriterResult<T> = Result<T, WriterError>;

impl From<serde_json::Error> for WriterError {
    #[inline]
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::ReaderError;
    use std::error::Error;
    use std::io;

    #[test]
    fn display_formats_variants() {
        let io_err = ReaderError::Io(io::Error::other("oops"));
        assert!(format!("{io_err}").contains("oops"));

        let st_err = ReaderError::SafeTensors(safetensors::SafeTensorError::TensorNotFound(
            "missing".into(),
        ));
        assert!(format!("{st_err}").contains("SafeTensors error"));

        let sllm_err = ReaderError::ServerlessLlm("bad".into());
        assert!(format!("{sllm_err}").contains("ServerlessLLM format error"));
    }

    #[test]
    fn from_conversions_set_sources() {
        let io_error = io::Error::other("source");
        let reader_error: ReaderError = io_error.into();
        assert!(reader_error.source().is_some());

        let st_error: ReaderError =
            safetensors::SafeTensorError::TensorNotFound("missing".into()).into();
        assert!(st_error.source().is_some());
    }
}
