//! High-performance I/O backends for tensor storage.
//!
//! This module provides zero-copy I/O operations optimized for large tensor files.
//! The API is impl-based: readers and writers are concrete types with methods.
//!
//! # Default Async Backend
//!
//! `AsyncReader` and `AsyncWriter` are backed by Tokio on all platforms, providing
//! portable async I/O. They are lightweight wrappers around ` TokioReader` and
//! `TokioWriter` respectively.
//!
//! # Explicit io_uring Backend
//!
//! On Linux, an explicit `io_uring`-based backend is available at `backends::io_uring`.
//! It exposes `Reader` and `Writer` types for specialized Linux workloads. Use it
//! directly when you need Linux-specific io_uring behavior.
//!
//! # Sync Operations
//!
//! `SyncReader` and `SyncWriter` use memory-mapped I/O on Linux and standard file
//! I/O elsewhere.
//!
//! # Usage
//!
//! ```rust,ignore
//! use tensor_store::backends::{AsyncReader, AsyncWriter, SyncReader};
//! use std::path::Path;
//!
//! // Tokio-backed async (portable)
//! let mut reader = AsyncReader::new();
//! let data = reader.load(Path::new("model.safetensors")).await?;
//!
//! // Async writer - stateful and path-bound
//! let mut writer = AsyncWriter::create(Path::new("output.bin")).await?;
//! writer.write_at(0, data).await?;
//! writer.sync_all().await?;
//!
//! // Sync reader
//! let mut sync_reader = SyncReader::new();
//! let chunk = sync_reader.load_range(Path::new("model.safetensors"), 1024, 512)?;
//!
//! // Explicit io_uring backend (Linux only)
//! #[cfg(target_os = "linux")]
//! {
//!     use tensor_store::backends::io_uring::Reader as IoUringReader;
//!     let mut reader = IoUringReader::new();
//!     let data = reader.load(Path::new("model.safetensors"))?;
//! }
//! ```

mod async_io;
pub mod batch;
pub mod buffer_slice;
pub mod byte;
#[cfg(target_os = "linux")]
pub mod io_uring;
pub mod mmap;
#[cfg(target_os = "linux")]
mod odirect;
mod sync_io;

use std::path::PathBuf;
use zeropool::BufferPool;

const CHECKPOINT_NUM_SHARDS: usize = 8;
const CHECKPOINT_TLS_CACHE_SIZE: usize = 4;
const CHECKPOINT_MAX_BUFFERS_PER_SHARD: usize = 32;
const CHECKPOINT_MIN_BUFFER_SIZE: usize = 1024 * 1024;

pub const MAX_SINGLE_READ: usize = 512 * 1024 * 1024;
pub const MAX_CHUNK_SIZE: usize = 128 * 1024 * 1024;
pub const MIN_CHUNK_SIZE: usize = 32 * 1024 * 1024;

#[inline]
pub fn calculate_chunks(file_size: usize) -> usize {
    if file_size == 0 {
        return 1;
    }
    file_size.div_ceil(MAX_CHUNK_SIZE).max(1)
}

/// Desired parallelism for submitting chunked I/O.
/// This is separate from chunk sizing, which is governed by MAX_CHUNK_SIZE.
#[inline]
pub fn chunk_budget() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .max(1)
}

pub struct ChunkPlan {
    pub offset: u64,
    pub len: usize,
}

pub fn build_chunk_plan(file_size: usize) -> Vec<ChunkPlan> {
    let chunks = calculate_chunks(file_size);
    let chunk_size = file_size.div_ceil(chunks);
    let mut plan = Vec::with_capacity(chunks);
    
    for i in 0..chunks {
        let start = i * chunk_size;
        if start >= file_size {
            break;
        }
        let end = std::cmp::min(start + chunk_size, file_size);
        let len = end - start;
        if len == 0 {
            break;
        }
        plan.push(ChunkPlan {
            offset: start as u64,
            len,
        });
    }
    plan
}

#[inline]
pub fn validate_read_count(actual: usize, expected: usize) -> IoResult<()> {
    if actual < expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected to read {expected} bytes, but read {actual}"),
        ));
    }
    Ok(())
}

static BUFFER_POOL: std::sync::OnceLock<BufferPool> = std::sync::OnceLock::new();

pub type BatchRequest = (PathBuf, u64, usize);

pub use std::io::Result as IoResult;

pub fn get_buffer_pool() -> &'static BufferPool {
    BUFFER_POOL.get_or_init(|| {
        BufferPool::builder()
            .num_shards(CHECKPOINT_NUM_SHARDS)
            .tls_cache_size(CHECKPOINT_TLS_CACHE_SIZE)
            .max_buffers_per_shard(CHECKPOINT_MAX_BUFFERS_PER_SHARD)
            .min_buffer_size(CHECKPOINT_MIN_BUFFER_SIZE)
            .pinned_memory(true)
            .build()
    })
}

pub struct AsyncReader {
    inner: async_io::TokioReader,
}

impl AsyncReader {
    pub fn new() -> Self {
        Self { inner: async_io::TokioReader::new() }
    }

    pub async fn load(&mut self, path: impl AsRef<std::path::Path> + Send) -> IoResult<byte::OwnedBytes> {
        self.inner.load(path).await
    }

    pub async fn load_batch(&mut self, paths: &[PathBuf]) -> IoResult<Vec<byte::OwnedBytes>> {
        self.inner.load_batch(paths).await
    }

    pub async fn load_range(&mut self, path: impl AsRef<std::path::Path> + Send, offset: u64, len: usize) -> IoResult<byte::OwnedBytes> {
        self.inner.load_range(path, offset, len).await
    }

    pub async fn load_range_batch(&mut self, requests: &[BatchRequest]) -> IoResult<Vec<batch::FlattenedResult>> {
        self.inner.load_range_batch(requests).await
    }
}

impl Default for AsyncReader {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SyncReader {
    inner: sync_io::SyncReaderEngine,
}

impl SyncReader {
    pub fn new() -> Self {
        Self { inner: sync_io::SyncReaderEngine::new() }
    }

    pub fn load(&mut self, path: impl AsRef<std::path::Path>) -> IoResult<byte::OwnedBytes> {
        self.inner.load(path)
    }

    pub fn load_batch(&mut self, paths: &[PathBuf]) -> IoResult<Vec<byte::OwnedBytes>> {
        self.inner.load_batch(paths)
    }

    pub fn load_range(&mut self, path: impl AsRef<std::path::Path>, offset: u64, len: usize) -> IoResult<byte::OwnedBytes> {
        self.inner.load_range(path, offset, len)
    }

    pub fn load_range_batch(&mut self, requests: &[BatchRequest]) -> IoResult<Vec<batch::FlattenedResult>> {
        self.inner.load_range_batch(requests)
    }
}

impl Default for SyncReader {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AsyncWriter {
    inner: async_io::TokioWriter,
}

impl AsyncWriter {
    pub async fn create(path: impl AsRef<std::path::Path>) -> IoResult<Self> {
        let inner = async_io::TokioWriter::create(path.as_ref()).await?;
        Ok(Self { inner })
    }

    pub async fn write_all(&mut self, data: &[u8]) -> IoResult<()> {
        self.inner.write_all(data).await
    }

    pub async fn write_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()> {
        self.inner.write_at(offset, data).await
    }

    pub async fn sync_all(&mut self) -> IoResult<()> {
        self.inner.sync_all().await
    }
}

pub struct SyncWriter {
    inner: sync_io::SyncWriterEngine,
}

impl SyncWriter {
    pub fn create(path: impl AsRef<std::path::Path>) -> IoResult<Self> {
        let inner = sync_io::SyncWriterEngine::create(path.as_ref())?;
        Ok(Self { inner })
    }

    pub fn write_all(&mut self, data: &[u8]) -> IoResult<()> {
        self.inner.write_all(data)
    }

    pub fn write_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()> {
        self.inner.write_at(offset, data)
    }

    pub fn sync_all(&mut self) -> IoResult<()> {
        self.inner.sync_all()
    }
}

const PARALLELISM_TARGET_BYTES: f64 = 128.0 * 1024.0 * 1024.0;

pub(crate) fn bounded_async_concurrency(item_count: usize, total_bytes: u64, max_item_bytes: u64) -> usize {
    if item_count <= 1 {
        return 1;
    }

    let total = total_bytes.max(1) as f64;
    let max_item = max_item_bytes.max(1) as f64;
    let avg_item = total / item_count as f64;

    let count_factor = (item_count as f64).ln_1p();
    let size_factor = (PARALLELISM_TARGET_BYTES / avg_item).sqrt().clamp(0.5, 4.0);
    let skew_factor = (max_item / avg_item).sqrt().clamp(1.0, 4.0);

    let raw = 1.0 + count_factor * size_factor / skew_factor;
    let unbounded = raw.clamp(1.0, item_count as f64);

    let cpu_count = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .max(2);

    let min_floor = item_count.clamp(2, 4).min(cpu_count);

    (unbounded as usize)
        .min(cpu_count)
        .max(min_floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reader_construction() {
        let _ = AsyncReader::new();
        let _ = SyncReader::new();
    }
}
