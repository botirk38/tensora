//! Traits for tensor writers.

use crate::writers::error::WriterResult;
use std::path::Path;

/// Trait for asynchronous tensor writers.
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
    async fn write(path: impl AsRef<Path>, data: &Self::Input) -> WriterResult<()>;
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
    fn write_sync(path: impl AsRef<Path>, data: &Self::Input) -> WriterResult<()>;
}
