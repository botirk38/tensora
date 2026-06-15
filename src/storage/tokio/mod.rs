//! Tokio async storage engine.
//!
//! [`TokioStorage`] provides inherent `async fn` methods for file and range
//! reads. It does **not** implement the synchronous [`ReadableStorage`] trait;
//! callers use the inherent methods directly.
//!
//! Write access is obtained by constructing a [`TokioWriter`] directly with
//! [`TokioWriter::create`].
//!
//! Internally, blocking I/O is offloaded to `tokio::task::spawn_blocking` so
//! that async tasks are never stalled on disk calls. On Linux the blocking
//! reads prefer O_DIRECT to bypass the page cache.
//!
//! [`ReadableStorage`]: crate::storage::ReadableStorage

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::{TokioStorage, TokioWriter};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::{TokioStorage, TokioWriter};

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("tensora storage::tokio currently has explicit implementations for Linux and macOS");
