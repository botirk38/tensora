//! Error types for tensor writers.

use std::io;

/// Errors that can occur during tensor writing operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
