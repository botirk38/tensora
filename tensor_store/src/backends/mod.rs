//! High-performance I/O backends for tensor storage.
//!
//! This module provides zero-copy I/O operations optimized for large tensor files.
//! The API is async-first with explicit sync alternatives for blocking contexts.
//!
//! # Platform Support
//!
//! - **Linux**: `io_uring` backend for async operations (maximum performance)
//! - **Cross-platform**: tokio async I/O fallback
//! - **Sync operations**: memory-mapped I/O on Linux, standard file I/O elsewhere
//!
//! # Usage
//!
//! ```rust,ignore
//! use tensor_store::backends;
//! use std::path::Path;
//!
//! // Async operations (default, recommended)
//! let data = backends::load("model.safetensors").await?;
//! let data = backends::load_parallel("model.safetensors", 4).await?;
//! backends::write_all("output.bin", data).await?;
//!
//! // Sync operations (for blocking contexts)
//! let data = backends::sync::load("model.safetensors")?;
//! let chunk = backends::sync::load_range("model.safetensors", 1024, 512)?;
//! ```

pub mod async_io;
pub mod batch;
pub mod buffer_slice;
pub mod io_uring;
pub mod mmap;
pub mod odirect;
pub mod sync_io;

pub use std::io::Result as IoResult;
use zeropool::BufferPool;

// Tuned for ML checkpoint loading (matches zeropool's profiling example).
const CHECKPOINT_NUM_SHARDS: usize = 8;
const CHECKPOINT_TLS_CACHE_SIZE: usize = 4;
const CHECKPOINT_MAX_BUFFERS_PER_SHARD: usize = 32;
const CHECKPOINT_MIN_BUFFER_SIZE: usize = 1024 * 1024; // 1MB minimum (drop tiny buffers)
/// Global buffer pool for tensor data.
/// Uses BufferPool for efficient buffer management.
static BUFFER_POOL: std::sync::OnceLock<BufferPool> = std::sync::OnceLock::new();

/// Build a buffer pool tuned for checkpoint loading workloads.
///
/// Settings mirror zeropool's ML checkpoint loader profile:
/// - Enough shards to reduce contention while keeping cache locality
/// - Modest TLS cache to keep metadata + weight buffers hot per thread
/// - Larger shard capacity to absorb bursty tensor allocations
/// - 1MB minimum buffer size to avoid polluting the pool with tiny buffers
fn build_buffer_pool() -> BufferPool {
    BufferPool::builder()
        .num_shards(CHECKPOINT_NUM_SHARDS)
        .tls_cache_size(CHECKPOINT_TLS_CACHE_SIZE)
        .max_buffers_per_shard(CHECKPOINT_MAX_BUFFERS_PER_SHARD)
        .min_buffer_size(CHECKPOINT_MIN_BUFFER_SIZE)
        // Pin memory to avoid page faults during high-throughput checkpoint reads.
        .pinned_memory(true)
        .build()
}

/// Get the global buffer pool instance.
#[inline]
pub fn get_buffer_pool() -> &'static BufferPool {
    BUFFER_POOL.get_or_init(build_buffer_pool)
}

// ============================================================================
// Async operations (default, platform-optimized)
// ============================================================================

/// Load entire file contents asynchronously.
///
/// Uses `io_uring` on Linux, tokio async I/O elsewhere.
#[cfg(target_os = "linux")]
pub use io_uring::load;

/// Load entire file contents asynchronously.
///
/// Uses tokio async I/O (fallback for non-Linux).
#[cfg(not(target_os = "linux"))]
pub use async_io::load;

/// Load file in parallel chunks asynchronously.
///
/// Uses `io_uring` on Linux, tokio async I/O elsewhere.
#[cfg(target_os = "linux")]
pub use io_uring::load_parallel;

#[cfg(not(target_os = "linux"))]
pub use async_io::load_parallel;

/// Load a specific byte range from file asynchronously.
///
/// Uses `io_uring` on Linux, tokio async I/O elsewhere.
#[cfg(target_os = "linux")]
pub use io_uring::load_range;

#[cfg(not(target_os = "linux"))]
pub use async_io::load_range;

/// Load multiple byte ranges from files asynchronously.
///
/// Uses `io_uring` on Linux, tokio async I/O elsewhere.
#[cfg(target_os = "linux")]
pub use io_uring::load_batch;

/// Load multiple byte ranges from files asynchronously.
///
/// Uses tokio async I/O (fallback for non-Linux).
#[cfg(not(target_os = "linux"))]
pub use async_io::load_batch;

/// Write entire buffer to file asynchronously.
///
/// Uses `io_uring` on Linux, tokio async I/O elsewhere.
/// Write entire buffer to file asynchronously.
#[cfg(target_os = "linux")]
pub use io_uring::write_all;

/// Write entire buffer to file asynchronously (tokio fallback).
#[cfg(not(target_os = "linux"))]
pub use async_io::write_all;

/// Synchronous blocking I/O
pub mod sync {
    pub use super::sync_io::{load, load_range, load_range_batch, write_all};
}
