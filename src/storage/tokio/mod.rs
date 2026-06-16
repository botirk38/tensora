//! Tokio async storage engine.
//!
//! All platforms share the same implementation in `shared.rs` — each method
//! offloads to the platform [`SyncStorage`] via `spawn_blocking`.  The
//! per-platform files (`linux.rs`, `macos.rs`, `windows.rs`) are thin
//! re-exports kept so callers can continue to write
//! `use tensora::storage::tokio::TokioStorage`.
//!
//! [`SyncStorage`]: crate::storage::sync::SyncStorage
//! [`AsyncReadableStorage`]: crate::storage::AsyncReadableStorage
//! [`AsyncWritableStorage`]: crate::storage::AsyncWritableStorage

mod shared;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::TokioStorage;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::TokioStorage;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::TokioStorage;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
compile_error!("tensora storage::tokio supports Linux, macOS, and Windows only");
