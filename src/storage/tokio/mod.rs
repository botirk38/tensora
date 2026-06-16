//! Tokio async storage engine.
//!
//! [`TokioStorage`] implements [`AsyncReadableStorage`] and [`AsyncWritableStorage`].
//! Each OS has an explicit implementation:
//! - Linux: O_DIRECT-aware reads via spawn_blocking; positioned async writes.
//! - macOS: spawn_blocking over macOS SyncStorage reads; positioned async writes.
//! - Windows: spawn_blocking over Windows SyncStorage reads; positioned async writes.
//!
//! [`AsyncReadableStorage`]: crate::storage::AsyncReadableStorage
//! [`AsyncWritableStorage`]: crate::storage::AsyncWritableStorage

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
