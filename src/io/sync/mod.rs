//! Synchronous blocking I/O backend.
//!
//! [`Sync`] implements [`BlockingIo`](crate::io::BlockingIo).
//! Each OS has an explicit implementation:
//! - Linux: O_DIRECT-aware chunked reads with `write_at` positioned writes.
//! - macOS: `std::os::unix::fs::FileExt` positioned I/O.
//! - Windows: `std::os::windows::fs::FileExt` positioned I/O.

/// Options for the synchronous I/O backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SyncOptions {
    /// Number of Rayon worker threads for batch reads/writes.
    ///
    /// `None` uses the global Rayon thread pool. `Some(n)` creates a backend-local
    /// pool with exactly `n` threads; `n == 0` is rejected by [`Sync::with_options`].
    pub batch_threads: Option<usize>,
}

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::Sync;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::Sync;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::Sync;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
compile_error!("tensora io::sync supports Linux, macOS, and Windows only");
