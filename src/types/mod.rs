//! Shared types, traits, and errors for tensor operations.
//!
//! This module contains the common types and traits used across all format modules.

pub mod error;
pub mod traits;

pub use error::{ReaderError, ReaderResult, WriterError, WriterResult};
pub use traits::{AsyncReader, AsyncWriter, SyncReader, SyncWriter, TensorMetadata};
