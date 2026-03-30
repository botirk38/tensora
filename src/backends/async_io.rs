//! Portable async I/O backend using Tokio (non-Linux platforms).

use super::{BatchRequest, IoResult, batch::{FlattenedResult, group_requests_by_file}, get_buffer_pool};
use std::path::Path;
use std::sync::Arc;

pub(crate) struct TokioReader;

impl TokioReader {
    pub(crate) const fn new() -> Self {
        Self
    }

    pub(crate) async fn load(&mut self, path: impl AsRef<Path> + Send) -> IoResult<Vec<u8>> {
        let path_buf = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || {
            use std::fs::File;
            use std::io::Read;
            let mut file = File::open(&path_buf)?;
            let len = usize::try_from(file.metadata()?.len())
                .map_err(|_| std::io::Error::other("file too large"))?;
            if len == 0 {
                return Ok(Vec::new());
            }
            let mut buf = get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(buf.into_inner())
        })
        .await
        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    pub(crate) async fn load_range(&mut self, path: impl AsRef<Path> + Send, offset: u64, len: usize) -> IoResult<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let path_buf = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || {
            use std::fs::File;
            use std::io::{Read, Seek, SeekFrom};
            let mut file = File::open(&path_buf)?;
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(buf.into_inner())
        })
        .await
        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    pub(crate) async fn load_range_batch(&mut self, requests: &[BatchRequest]) -> IoResult<Vec<FlattenedResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);
        let mut handles: Vec<_> = Vec::with_capacity(requests.len());

        for (path, reqs) in grouped {
            for req in reqs {
                let path_buf = path.clone();
                let handle = tokio::task::spawn_blocking(move || -> std::io::Result<(usize, Arc<[u8]>, usize, usize)> {
                    let mut file = std::fs::File::open(&path_buf)?;
                    use std::io::{Read, Seek};
                    file.seek(std::io::SeekFrom::Start(req.offset))?;
                    let mut buf = get_buffer_pool().get(req.len);
                    Read::read_exact(&mut file, &mut buf[..])?;
                    let idx = req.idx;
                    let len = req.len;
                    let data: Arc<[u8]> = buf.into_inner().into();
                    Ok((idx, data, 0, len))
                });
                handles.push(handle);
            }
        }

        let mut indexed: Vec<(usize, Arc<[u8]>, usize, usize)> = Vec::with_capacity(requests.len());
        for handle in handles {
            let result = handle
                .await
                .map_err(|_| std::io::Error::other("spawn_blocking panicked"))??;
            indexed.push(result);
        }

        indexed.sort_by_key(|(idx, _, _, _)| *idx);
        Ok(indexed.into_iter().map(|(_, data, offset, len)| (data, offset, len)).collect())
    }
}

pub(crate) struct TokioWriter {
    file: tokio::fs::File,
}

impl std::fmt::Debug for TokioWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokioWriter").finish_non_exhaustive()
    }
}

impl TokioWriter {
    pub(crate) async fn create(path: &Path) -> IoResult<Self> {
        if let Some(parent) = path.parent() && !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = tokio::fs::File::create(path).await?;
        Ok(Self { file })
    }

    pub(crate) async fn write_all(&mut self, data: Vec<u8>) -> IoResult<()> {
        use tokio::io::{AsyncSeekExt, AsyncWriteExt};
        self.file.set_len(0).await?;
        self.file.seek(std::io::SeekFrom::Start(0)).await?;
        self.file.write_all(&data).await
    }

    pub(crate) async fn write_at(&mut self, offset: u64, data: Vec<u8>) -> IoResult<()> {
        use tokio::io::{AsyncSeekExt, AsyncWriteExt};
        self.file.seek(std::io::SeekFrom::Start(offset)).await?;
        self.file.write_all(&data).await
    }

    pub(crate) async fn sync_all(&mut self) -> IoResult<()> {
        self.file.sync_all().await
    }
}