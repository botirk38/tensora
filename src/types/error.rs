//! Error types for tensor readers and writers.

use std::fmt;
use std::io;

// ============================================================================
// Reader Errors
// ============================================================================

/// Unified error type for all tensor readers.
#[derive(Debug)]
pub enum ReaderError {
    /// I/O error occurred during reading.
    Io(io::Error),

    /// `SafeTensors` format error.
    SafeTensors(safetensors::SafeTensorError),

    /// `ServerlessLLM` format error (invalid JSON or structure).
    ServerlessLlm(String),

    /// `TensorStore` format error.
    TensorStore(String),

    /// Invalid tensor metadata.
    InvalidMetadata(String),
}

impl fmt::Display for ReaderError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::SafeTensors(err) => write!(f, "SafeTensors error: {err}"),
            Self::ServerlessLlm(msg) => write!(f, "ServerlessLLM format error: {msg}"),
            Self::TensorStore(msg) => write!(f, "TensorStore format error: {msg}"),
            Self::InvalidMetadata(msg) => write!(f, "Invalid tensor metadata: {msg}"),
        }
    }
}

impl std::error::Error for ReaderError {
    #[inline]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::SafeTensors(err) => Some(err),
            Self::ServerlessLlm(_) | Self::TensorStore(_) | Self::InvalidMetadata(_) => None,
        }
    }
}

impl From<io::Error> for ReaderError {
    #[inline]
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<safetensors::SafeTensorError> for ReaderError {
    #[inline]
    fn from(err: safetensors::SafeTensorError) -> Self {
        Self::SafeTensors(err)
    }
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
    use super::*;
    use std::error::Error;

    #[test]
    fn display_formats_variants() {
        let io_err = ReaderError::Io(io::Error::new(io::ErrorKind::Other, "oops"));
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
        let io_error = io::Error::new(io::ErrorKind::Other, "source");
        let reader_error: ReaderError = io_error.into();
        assert!(reader_error.source().is_some());

        let st_error: ReaderError =
            safetensors::SafeTensorError::TensorNotFound("missing".into()).into();
        assert!(st_error.source().is_some());
    }
}
