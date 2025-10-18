use parking_lot::Mutex;
use std::cell::RefCell;
use std::sync::Arc;

/// Default minimum buffer size (1MB) - buffers smaller than this won't be kept
/// Rationale: Tensor files are typically large (100MB+), small buffers aren't worth pooling
const DEFAULT_MIN_BUFFER_SIZE: usize = 1024 * 1024; // 1MB

/// Maximum number of buffers to keep in the pool to prevent unbounded growth
/// Rationale: Limit memory usage. With 500MB buffers, 16 = 8GB max pool size
const DEFAULT_MAX_POOL_SIZE: usize = 16;

thread_local! {
    /// Thread-local cache for lock-free fast path in single-threaded scenarios
    static TLS_BUFFER: RefCell<Option<Vec<u8>>> = RefCell::new(None);
}

/// A highly optimized thread-safe pool for buffer reuse
///
/// Optimizations:
/// - Thread-local cache for lock-free fast path
/// - Uses parking_lot::Mutex (faster than std::sync::Mutex)
/// - First-fit allocation (O(1) instead of O(n) best-fit)
/// - Minimizes lock scope
/// - Uses swap_remove for O(1) removal
/// - Inlines hot paths
/// - Unsafe set_len to avoid zero-fill overhead (critical for 500MB buffers)
#[derive(Clone)]
pub struct PinnedPool {
    buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    min_size: usize,
    max_pool_size: usize,
}

impl PinnedPool {
    /// Create a new buffer pool with default configuration
    #[inline]
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_MIN_BUFFER_SIZE, DEFAULT_MAX_POOL_SIZE)
    }

    /// Create a pool with full configuration control
    pub fn with_limits(min_size: usize, max_pool_size: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(Vec::new())),
            min_size,
            max_pool_size,
        }
    }

    /// Get a buffer of at least the specified size from the pool
    /// Returns an owned Vec for maximum compatibility (io-uring, tokio, etc.)
    #[inline]
    pub fn get(&self, size: usize) -> Vec<u8> {
        // Fastest path: thread-local cache (lock-free)
        let tls_hit = TLS_BUFFER.with(|tls| {
            let mut cache = tls.borrow_mut();
            if let Some(mut buf) = cache.take() {
                if buf.capacity() >= size {
                    unsafe {
                        // SAFETY: capacity >= size is checked above
                        // This avoids the expensive zero-fill from resize()
                        buf.set_len(size);
                    }
                    return Some(buf);
                }
                // Buffer too small, return to shared pool
                *cache = Some(buf);
            }
            None
        });

        if let Some(buf) = tls_hit {
            return buf;
        }

        // Fast path: try shared pool
        let mut buffers = self.buffers.lock();

        // First-fit: take first buffer that's large enough (O(1) average case)
        let idx_opt = buffers.iter().position(|b| b.capacity() >= size);

        if let Some(idx) = idx_opt {
            let mut buffer = buffers.swap_remove(idx);
            drop(buffers); // Release lock early

            // Reuse existing buffer
            unsafe {
                // SAFETY: capacity >= size is guaranteed by position() filter
                // This avoids the expensive zero-fill from resize()
                buffer.set_len(size);
            }
            buffer
        } else {
            drop(buffers); // Release lock before allocation
            // Allocate new buffer
            vec![0u8; size]
        }
    }

    /// Return a buffer to the pool for reuse
    /// Call this after you're done with a buffer to enable reuse
    #[inline]
    pub fn put(&self, mut buffer: Vec<u8>) {
        // Clear length but keep capacity
        buffer.clear();

        // Fastest path: return to thread-local cache (lock-free)
        let can_store_tls = TLS_BUFFER.with(|tls| tls.borrow().is_none());

        if can_store_tls {
            TLS_BUFFER.with(|tls| {
                *tls.borrow_mut() = Some(buffer);
            });
            return;
        }

        // Fall back to shared pool
        if buffer.capacity() < self.min_size {
            return; // Too small, don't keep
        }

        let mut buffers = self.buffers.lock();
        // Limit pool size to prevent unbounded growth
        if buffers.len() < self.max_pool_size {
            buffers.push(buffer);
        }
    }

    /// Pre-allocate buffers in the pool
    pub fn preallocate(&self, count: usize, size: usize) {
        let mut buffers = self.buffers.lock();
        buffers.reserve(count);
        for _ in 0..count {
            buffers.push(Vec::with_capacity(size.max(self.min_size)));
        }
    }

    /// Get the number of buffers currently in the pool
    #[inline]
    pub fn len(&self) -> usize {
        self.buffers.lock().len()
    }

    /// Check if the pool is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffers.lock().is_empty()
    }
}

impl Default for PinnedPool {
    fn default() -> Self {
        Self::new()
    }
}
