use std::io::Result as IoResult;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod loaders;

#[cfg(target_os = "linux")]
pub async fn load_safetensors(
    path: &str,
    pool: Option<&zeropool::BufferPool>,
) -> IoResult<Vec<u8>> {
    loaders::uring::BasicLoader::load(path, pool).await
}

#[cfg(not(target_os = "linux"))]
pub async fn load_safetensors(
    path: &str,
    pool: Option<&zeropool::BufferPool>,
) -> IoResult<Vec<u8>> {
    loaders::tokio::BasicLoader::load(path, pool).await
}
