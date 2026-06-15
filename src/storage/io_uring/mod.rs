//! io_uring storage engine (Linux only).
//!
//! [`IoUringStorage`] implements [`ReadableStorage`] and [`StorageEngine`].
//! It uses the kernel's io_uring interface for high-throughput, low-latency
//! batch reads via a persistent submission/completion ring, with optional
//! SQ polling and O_DIRECT support.
//!
//! Write access is obtained by constructing an [`IoUringWriter`] directly with
//! [`IoUringWriter::create`].
//!
//! io_uring is only available on Linux; this module is entirely
//! `#[cfg(target_os = "linux")]`.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::{IoUringStorage, IoUringWriter};
