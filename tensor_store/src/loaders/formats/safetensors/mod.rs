use crate::loaders::backends;

/// Load tensor data using the appropriate backend for the current platform
#[inline]
pub async fn load(path: &str) -> backends::IoResult<Vec<u8>> {
    #[cfg(target_os = "linux")]
    {
        backends::io_uring::load(path).await
    }
    #[cfg(not(target_os = "linux"))]
    {
        backends::async_io::load(path).await
    }
}

/// Load tensor data in parallel chunks using the appropriate backend
#[inline]
pub async fn load_parallel(path: &str, chunks: usize) -> backends::IoResult<Vec<u8>> {
    #[cfg(target_os = "linux")]
    {
        backends::io_uring::load_parallel(path, chunks).await
    }
    #[cfg(not(target_os = "linux"))]
    {
        backends::async_io::load_parallel(path, chunks).await
    }
}

/// Load a specific byte range from tensor data using the appropriate backend
#[inline]
pub async fn load_range(path: &str, offset: u64, len: usize) -> backends::IoResult<Vec<u8>> {
    #[cfg(target_os = "linux")]
    {
        backends::io_uring::load_range(path, offset, len).await
    }
    #[cfg(not(target_os = "linux"))]
    {
        backends::async_io::load_range(path, offset, len).await
    }
}
