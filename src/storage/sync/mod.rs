//! Synchronous blocking storage engine.
//!
//! [`SyncStorage`] implements [`ReadableStorage`] and [`WritableStorage`].
//! Each OS has an explicit implementation:
//! - Linux: O_DIRECT-aware chunked reads with `write_at` positioned writes.
//! - macOS: `std::os::unix::fs::FileExt` positioned I/O.
//! - Windows: `std::os::windows::fs::FileExt` positioned I/O.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::SyncStorage;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::SyncStorage;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::SyncStorage;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
compile_error!("tensora storage::sync supports Linux, macOS, and Windows only");
