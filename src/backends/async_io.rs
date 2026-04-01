//! Portable async I/O backend using Tokio (non-Linux platforms).

use super::{
    BackendKind, BatchRequest, IoResult,
    batch::{
        BatchResult, FlattenedResult, coalesce_requests, flatten_results, group_requests_by_file,
    },
    byte::OwnedBytes,
    file_chunk_plan, get_buffer_pool, range_batch_plan,
};
use std::path::Path;
use std::sync::Arc;

pub(crate) struct TokioReader;

impl TokioReader {
    pub(crate) const fn new() -> Self {
        Self
    }

    pub(crate) async fn load(&mut self, path: impl AsRef<Path> + Send) -> IoResult<OwnedBytes> {
        let path_buf = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || {
            use std::fs::File;
            use std::io::Read;
            let mut file = File::open(&path_buf)?;
            let len = usize::try_from(file.metadata()?.len())
                .map_err(|_| std::io::Error::other("file too large"))?;
            if len == 0 {
                return Ok(OwnedBytes::Shared(Arc::new([])));
            }
            let mut buf = get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(OwnedBytes::Pooled(buf))
        })
        .await
        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    pub(crate) async fn load_batch(
        &mut self,
        paths: &[std::path::PathBuf],
    ) -> IoResult<Vec<OwnedBytes>> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }

        let total_bytes: usize = paths
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .filter_map(|meta| usize::try_from(meta.len()).ok())
            .sum();
        let concurrency = file_chunk_plan(total_bytes.max(1), BackendKind::Async)
            .target_inflight
            .min(paths.len())
            .max(1);

        let mut results = Vec::with_capacity(paths.len());
        for chunk in paths.chunks(concurrency) {
            let mut handles: Vec<tokio::task::JoinHandle<std::io::Result<OwnedBytes>>> =
                Vec::with_capacity(chunk.len());
            for path in chunk {
                let path = path.clone();
                let handle = tokio::task::spawn_blocking(move || {
                    use std::fs::File;
                    use std::io::Read;

                    let mut file = File::open(&path)?;
                    let len = usize::try_from(file.metadata()?.len())
                        .map_err(|_| std::io::Error::other("file too large"))?;
                    if len == 0 {
                        return Ok(OwnedBytes::Shared(Arc::new([])));
                    }
                    let mut buf = get_buffer_pool().get(len);
                    file.read_exact(&mut buf[..])?;
                    Ok(OwnedBytes::from_pooled(buf))
                });
                handles.push(handle);
            }

            for handle in handles {
                results.push(
                    handle
                        .await
                        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))??,
                );
            }
        }
        Ok(results)
    }

    pub(crate) async fn load_range(
        &mut self,
        path: impl AsRef<Path> + Send,
        offset: u64,
        len: usize,
    ) -> IoResult<OwnedBytes> {
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let path_buf = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || {
            use std::fs::File;
            use std::io::{Read, Seek, SeekFrom};
            let mut file = File::open(&path_buf)?;
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(OwnedBytes::Pooled(buf))
        })
        .await
        .map_err(|_| std::io::Error::other("spawn_blocking panicked"))?
    }

    pub(crate) async fn load_range_batch(
        &mut self,
        requests: &[BatchRequest],
    ) -> IoResult<Vec<FlattenedResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);
        let total_bytes: usize = requests.iter().map(|(_, _, len)| *len).sum();
        let batch_plan = range_batch_plan(requests.len(), total_bytes, BackendKind::Async);
        let grouped_vec = coalesce_requests(grouped, batch_plan.coalesce_window_bytes);
        let concurrency = batch_plan.target_inflight.min(grouped_vec.len()).max(1);

        let mut grouped_results: Vec<Vec<BatchResult>> = Vec::with_capacity(grouped_vec.len());
        for chunk in grouped_vec.chunks(concurrency) {
            let mut handles: Vec<_> = Vec::with_capacity(chunk.len());
            for group in chunk {
                let path_buf = group.path.clone();
                let offset = group.offset;
                let len = group.len;
                let members = group.members.clone();
                let handle = tokio::task::spawn_blocking(
                    move || -> std::io::Result<Vec<BatchResult>> {
                        use std::io::{Read, Seek};
                        let mut file = std::fs::File::open(&path_buf)?;
                        file.seek(std::io::SeekFrom::Start(offset))?;
                        let mut buf = get_buffer_pool().get(len);
                        Read::read_exact(&mut file, &mut buf[..])?;
                        let backing: Arc<[u8]> = buf.into_inner().into();
                        let mut results = Vec::with_capacity(members.len());
                        for member in members {
                            let start = member.relative_offset;
                            let end = start.saturating_add(member.len);
                            let slice = backing.get(start..end).ok_or_else(|| {
                                std::io::Error::new(
                                    std::io::ErrorKind::UnexpectedEof,
                                    format!(
                                        "coalesced slice out of bounds for request {}: {}..{} of {}",
                                        member.idx,
                                        start,
                                        end,
                                        backing.len()
                                    ),
                                )
                            })?;
                            results.push((member.idx, Arc::<[u8]>::from(slice), 0, member.len));
                        }
                        Ok(results)
                    },
                );
                handles.push(handle);
            }

            for handle in handles {
                let results = handle
                    .await
                    .map_err(|_| std::io::Error::other("spawn_blocking panicked"))??;
                grouped_results.push(results);
            }
        }

        Ok(flatten_results(grouped_results))
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
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = tokio::fs::File::create(path).await?;
        Ok(Self { file })
    }

    pub(crate) async fn write_all(&mut self, data: &[u8]) -> IoResult<()> {
        use tokio::io::{AsyncSeekExt, AsyncWriteExt};
        self.file.set_len(0).await?;
        self.file.seek(std::io::SeekFrom::Start(0)).await?;
        self.file.write_all(data).await
    }

    pub(crate) async fn write_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()> {
        use tokio::io::{AsyncSeekExt, AsyncWriteExt};
        self.file.seek(std::io::SeekFrom::Start(offset)).await?;
        self.file.write_all(data).await
    }

    pub(crate) async fn sync_all(&mut self) -> IoResult<()> {
        self.file.sync_all().await
    }
}
