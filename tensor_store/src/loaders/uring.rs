use super::IoResult;
use crate::PinnedPool;
use tokio_uring::fs::File as UringFile;

pub struct BasicLoader;

impl BasicLoader {
    pub async fn load(path: &str, pool: Option<&PinnedPool>) -> IoResult<Vec<u8>> {
        let file = UringFile::open(path).await?;
        let metadata = std::fs::metadata(path)?;
        let file_size = metadata.len() as usize;

        // Get buffer from pool if available, otherwise allocate
        let buf = if let Some(pool) = pool {
            pool.get(file_size)
        } else {
            vec![0u8; file_size]
        };

        let (res, buf) = file.read_at(buf, 0).await;
        let n = res?;

        if n != file_size {
            file.close().await?;
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("Expected to read {} bytes, but read {}", file_size, n),
            ));
        }

        file.close().await?;

        Ok(buf)
    }
}
