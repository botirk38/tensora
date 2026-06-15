//! Synchronous blocking storage engine.
//!
//! [`SyncStorage`] implements [`ReadableStorage`] and [`StorageEngine`].
//! On Linux it uses O_DIRECT where possible; on macOS it uses buffered `std::fs` I/O.
//!
//! Write access is obtained by constructing a [`SyncWriter`] directly with
//! [`SyncWriter::create`].

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::{SyncStorage, SyncWriter};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::{SyncStorage, SyncWriter};

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("tensora storage::sync currently has explicit implementations for Linux and macOS");
