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
//! let backend = backends::async_backend();
//! let data = backend.load(Path::new("model.safetensors")).await?;
//! let data = backend
//!     .load_parallel(Path::new("model.safetensors"), 4)
//!     .await?;
//! backend.write_all(Path::new("output.bin"), data).await?;
//!
//! // Sync operations (for blocking contexts)
//! let sync_backend = backends::sync_backend();
//! let data = sync_backend.load(Path::new("model.safetensors"))?;
//! let chunk = sync_backend.load_range(Path::new("model.safetensors"), 1024, 512)?;
//! ```

pub mod async_io;
pub mod batch;
pub mod buffer_slice;
pub mod io_uring;
pub mod mmap;
pub mod odirect;
pub mod sync_io;

pub use std::io::Result as IoResult;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use zeropool::BufferPool;

// Tuned for ML checkpoint loading (matches zeropool's profiling example).
const CHECKPOINT_NUM_SHARDS: usize = 8;
const CHECKPOINT_TLS_CACHE_SIZE: usize = 4;
const CHECKPOINT_MAX_BUFFERS_PER_SHARD: usize = 32;
const CHECKPOINT_MIN_BUFFER_SIZE: usize = 1024 * 1024; // 1MB minimum (drop tiny buffers)
/// Global buffer pool for tensor data.
/// Uses BufferPool for efficient buffer management.
static BUFFER_POOL: std::sync::OnceLock<BufferPool> = std::sync::OnceLock::new();

/// Batch request type used by backend interfaces.
pub type BatchRequest = (PathBuf, u64, usize);

/// Boxed future alias to keep backend trait signatures readable.
pub type AsyncBackendFuture<'a, T> =
    Pin<Box<dyn Future<Output = IoResult<T>> + 'a>>;

/// Safe interface for asynchronous backends.
pub trait AsyncBackend: Send + Sync + 'static {
    fn load<'a>(&'a self, path: &'a Path) -> AsyncBackendFuture<'a, Vec<u8>>;
    fn load_parallel<'a>(
        &'a self,
        path: &'a Path,
        chunks: usize,
    ) -> AsyncBackendFuture<'a, Vec<u8>>;
    fn load_range<'a>(
        &'a self,
        path: &'a Path,
        offset: u64,
        len: usize,
    ) -> AsyncBackendFuture<'a, Vec<u8>>;
    fn load_batch<'a>(
        &'a self,
        requests: &'a [BatchRequest],
    ) -> AsyncBackendFuture<'a, Vec<batch::FlattenedResult>>;
    fn write_all<'a>(
        &'a self,
        path: &'a Path,
        data: Vec<u8>,
    ) -> AsyncBackendFuture<'a, ()>;
}

/// Safe interface for synchronous backends.
pub trait SyncBackend: Send + Sync + 'static {
    fn load(&self, path: &Path) -> IoResult<Vec<u8>>;
    fn load_parallel(&self, path: &Path, chunks: usize) -> IoResult<Vec<u8>>;
    fn load_range(&self, path: &Path, offset: u64, len: usize) -> IoResult<Vec<u8>>;
    fn load_range_batch(
        &self,
        requests: &[BatchRequest],
    ) -> IoResult<Vec<batch::FlattenedResult>>;
    fn write_all(&self, path: &Path, data: Vec<u8>) -> IoResult<()>;
}

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

#[cfg(target_os = "linux")]
static ASYNC_BACKEND: io_uring::IoUringBackend = io_uring::IoUringBackend;
#[cfg(not(target_os = "linux"))]
static ASYNC_BACKEND: async_io::TokioAsyncBackend = async_io::TokioAsyncBackend;
static SYNC_BACKEND: sync_io::DefaultSyncBackend = sync_io::DefaultSyncBackend;

/// Get the default asynchronous backend for the current platform.
pub fn async_backend() -> &'static dyn AsyncBackend {
    &ASYNC_BACKEND
}

/// Get the default synchronous backend for the current platform.
pub fn sync_backend() -> &'static dyn SyncBackend {
    &SYNC_BACKEND
}

#[cfg(test)]
mod tests {
    use super::{
        async_backend, get_buffer_pool, sync_backend, CHECKPOINT_MAX_BUFFERS_PER_SHARD,
        CHECKPOINT_MIN_BUFFER_SIZE, CHECKPOINT_NUM_SHARDS, CHECKPOINT_TLS_CACHE_SIZE,
    };

    #[test]
    fn test_buffer_pool_initialization() {
        let pool = get_buffer_pool();
        // Just verify we can get the pool without panicking
        assert!(!std::ptr::eq(pool, std::ptr::null()));
    }

    #[test]
    fn test_buffer_pool_singleton() {
        let pool1 = get_buffer_pool();
        let pool2 = get_buffer_pool();
        // Verify both references point to the same instance
        assert!(std::ptr::eq(pool1, pool2));
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_buffer_pool_config_values() {
        assert!(CHECKPOINT_NUM_SHARDS > 0);
        assert!(CHECKPOINT_NUM_SHARDS <= 64);
        assert!(CHECKPOINT_TLS_CACHE_SIZE > 0);
        assert!(CHECKPOINT_MAX_BUFFERS_PER_SHARD > 0);
        assert!(CHECKPOINT_MIN_BUFFER_SIZE >= 1024);
    }

    #[test]
    fn test_get_buffer_pool_multiple_calls() {
        // Call multiple times to verify stability
        for _ in 0..10 {
            let pool = get_buffer_pool();
            assert!(!std::ptr::eq(pool, std::ptr::null()));
        }
    }

    #[test]
    fn test_platform_specific_exports() {
        // Ensure singletons are stable and non-null
        let async_b1 = async_backend();
        let async_b2 = async_backend();
        assert!(std::ptr::eq(async_b1, async_b2));

        let sync_b1 = sync_backend();
        let sync_b2 = sync_backend();
        assert!(std::ptr::eq(sync_b1, sync_b2));
    }
}
