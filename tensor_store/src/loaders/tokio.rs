use super::IoResult;
use crate::PinnedPool;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncReadExt;

pub struct BasicLoader;

impl BasicLoader {
    pub async fn load(path: &str, pool: Option<&PinnedPool>) -> IoResult<Vec<u8>> {
        let mut file = TokioFile::open(path).await?;
        let metadata = file.metadata().await?;
        let file_size = metadata.len() as usize;

        // Get buffer from pool if available, otherwise allocate
        let mut buf = if let Some(pool) = pool {
            pool.get(file_size)
        } else {
            vec![0u8; file_size]
        };

        file.read_exact(&mut buf).await?;
        Ok(buf)
    }
}
