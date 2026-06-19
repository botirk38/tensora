//! [`Checkpoint`] trait — loading and saving format-specific checkpoints.

use crate::formats::error::{LoadResult, SaveResult};
use crate::formats::{AsyncBackend, Backend};
use std::future::Future;
use std::path::Path;

use super::Model;

/// Trait for loading and saving a format-specific checkpoint.
///
/// Implement this on format checkpoint types to provide loading (`load`,
/// `aload`, `open`) and saving (`save`, `asave`) paths. Loading produces
/// the format's [`Model`] type; saving consumes a checkpoint instance.
/// Sync and async variants should be semantically equivalent.
pub trait Checkpoint {
    /// The model type produced by this checkpoint.
    type Model: Model;

    /// Load a checkpoint eagerly using the chosen blocking backend.
    fn load(path: impl AsRef<Path>, backend: Backend) -> LoadResult<Self::Model>;

    /// Load a checkpoint eagerly using the chosen async backend.
    fn aload(
        path: impl AsRef<Path> + Send,
        backend: AsyncBackend,
    ) -> impl Future<Output = LoadResult<Self::Model>> + Send;

    /// Open a checkpoint lazily (e.g. via memory mapping).
    fn open(path: impl AsRef<Path>) -> LoadResult<Self::Model>;

    /// Write the checkpoint to `path` synchronously.
    fn save(&self, path: impl AsRef<Path>) -> SaveResult<()>;

    /// Write the checkpoint to `path` asynchronously.
    fn asave(&self, path: impl AsRef<Path> + Send) -> impl Future<Output = SaveResult<()>> + Send;
}
