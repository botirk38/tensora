//! Error types for tensor readers.

use std::fmt;
use std::io;

/// Unified error type for all tensor readers.
#[derive(Debug)]
#[non_exhaustive]
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
