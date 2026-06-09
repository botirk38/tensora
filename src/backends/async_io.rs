//! Async I/O backend using Tokio.
//!
//! On Linux, reads use O_DIRECT via `spawn_blocking` to bypass the page cache.

use super::{
    BackendKind, BatchRequest, IoResult,
    batch::{
        BatchResult, FlattenedResult, coalesce_requests, flatten_results, group_requests_by_file,
    },
    byte::OwnedBytes,
    file_chunk_plan, range_batch_plan,
};
use crate::backends::availability::BackendAvailability;
use std::path::Path;
use std::sync::Arc;

pub(crate) struct TokioReader;

impl TokioReader {
    pub(crate) const fn availability() -> BackendAvailability {
        BackendAvailability::Available
    }

    pub(crate) const fn new() -> Self {
        Self
    }

    pub(crate) async fn load(&mut self, path: impl AsRef<Path> + Send) -> IoResult<OwnedBytes> {
        let path_buf = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || Self::load_blocking(&path_buf))
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
                let handle = tokio::task::spawn_blocking(move || Self::load_blocking(&path));
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
        tokio::task::spawn_blocking(move || Self::load_range_blocking(&path_buf, offset, len))
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
                        let data = Self::load_range_blocking(&path_buf, offset, len)?;
                        let backing: Arc<[u8]> = data.into_shared();
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

    #[cfg(target_os = "linux")]
    fn load_blocking(path: &Path) -> IoResult<OwnedBytes> {
        use super::odirect::{AlignedBuffer, open_prefer_direct, read_direct, round_up_to_block};
        use std::io::Read;

        let (mut file, direct) = open_prefer_direct(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        if direct {
            let aligned_len = round_up_to_block(len);
            let mut buf = AlignedBuffer::new(aligned_len)?;
            buf.set_len(aligned_len);
            read_direct(&mut file, buf.as_mut_slice(), len)?;
            buf.set_len(len);
            Ok(OwnedBytes::Aligned(buf))
        } else {
            let mut buf = super::get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(OwnedBytes::Pooled(buf))
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn load_blocking(path: &Path) -> IoResult<OwnedBytes> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut buf = super::get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(OwnedBytes::Pooled(buf))
    }

    #[cfg(target_os = "linux")]
    fn load_range_blocking(path: &Path, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        use super::odirect::{
            BLOCK_SIZE_U64, AlignedBuffer, open_prefer_direct, read_direct, round_up_to_block,
        };
        use std::io::{Read, Seek, SeekFrom};

        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let (mut file, direct) = open_prefer_direct(path)?;
        if direct {
            let aligned_offset = offset & !(BLOCK_SIZE_U64 - 1);
            let head_skip = (offset - aligned_offset) as usize;
            let aligned_len = round_up_to_block(head_skip + len);

            file.seek(SeekFrom::Start(aligned_offset))?;
            let mut buf = AlignedBuffer::new(aligned_len)?;
            buf.set_len(aligned_len);
            read_direct(&mut file, buf.as_mut_slice(), head_skip + len)?;

            if head_skip == 0 {
                buf.set_len(len);
                return Ok(OwnedBytes::Aligned(buf));
            }
            let slice = &buf.as_slice()[head_skip..head_skip + len];
            Ok(OwnedBytes::Shared(Arc::from(slice)))
        } else {
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = super::get_buffer_pool().get(len);
            file.read_exact(&mut buf[..])?;
            Ok(OwnedBytes::Pooled(buf))
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn load_range_blocking(path: &Path, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        use std::io::{Read, Seek, SeekFrom};
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }
        let mut file = std::fs::File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = super::get_buffer_pool().get(len);
        file.read_exact(&mut buf[..])?;
        Ok(OwnedBytes::Pooled(buf))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::SyncReader;
    use tempfile::TempDir;

    fn write_file(path: &Path, len: usize, seed: u8) {
        let data: Vec<u8> = (0..len)
            .map(|i| seed.wrapping_add((i % 251) as u8))
            .collect();
        std::fs::write(path, data).unwrap();
    }

    #[tokio::test]
    async fn load_single_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("single.bin");
        write_file(&path, 1024, 42);

        let expected = std::fs::read(&path).unwrap();
        let mut reader = TokioReader::new();
        let actual = reader.load(&path).await.unwrap();
        assert_eq!(actual.as_ref(), &expected[..]);
    }

    #[tokio::test]
    async fn load_empty_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.bin");
        std::fs::write(&path, b"").unwrap();

        let mut reader = TokioReader::new();
        let actual = reader.load(&path).await.unwrap();
        assert!(actual.is_empty());
    }

    #[tokio::test]
    async fn load_batch_empty_vec() {
        let mut reader = TokioReader::new();
        let actual = reader.load_batch(&[]).await.unwrap();
        assert!(actual.is_empty());
    }

    #[tokio::test]
    async fn load_batch_matches_sync_reader() {
        let dir = TempDir::new().unwrap();
        let path_a = dir.path().join("a.bin");
        let path_b = dir.path().join("b.bin");
        write_file(&path_a, 2 * 1024 * 1024, 17);
        write_file(&path_b, 3 * 1024 * 1024, 23);

        let mut sync = SyncReader::new();
        let expected = sync.load_batch(&[path_a.clone(), path_b.clone()]).unwrap();

        let mut reader = TokioReader::new();
        let actual = reader.load_batch(&[path_a, path_b]).await.unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual_item, expected_item) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_item.as_ref(), expected_item.as_ref());
        }
    }

    #[tokio::test]
    async fn load_range_single() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("range.bin");
        write_file(&path, 1024, 7);
        let all = std::fs::read(&path).unwrap();

        let mut reader = TokioReader::new();
        let actual = reader.load_range(&path, 100, 200).await.unwrap();
        assert_eq!(actual.as_ref(), &all[100..300]);
    }

    #[tokio::test]
    async fn load_range_zero_len() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("range_zero.bin");
        write_file(&path, 1024, 7);

        let mut reader = TokioReader::new();
        let actual = reader.load_range(&path, 0, 0).await.unwrap();
        assert!(actual.is_empty());
    }

    #[tokio::test]
    async fn load_range_batch_empty() {
        let mut reader = TokioReader::new();
        let actual = reader.load_range_batch(&[]).await.unwrap();
        assert!(actual.is_empty());
    }

    #[tokio::test]
    async fn load_range_batch_matches_sync_reader() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ranges.bin");
        write_file(&path, 4 * 1024 * 1024, 11);

        let requests = vec![
            (path.clone(), 0, 256 * 1024),
            (path.clone(), 260 * 1024, 128 * 1024),
            (path.clone(), 2 * 1024 * 1024, 512 * 1024),
        ];

        let mut sync = SyncReader::new();
        let expected = sync.load_range_batch(&requests).unwrap();

        let mut reader = TokioReader::new();
        let actual = reader.load_range_batch(&requests).await.unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual_item, expected_item) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_item.1, expected_item.1);
            assert_eq!(actual_item.2, expected_item.2);
            assert_eq!(actual_item.0.as_ref(), expected_item.0.as_ref());
        }
    }

    #[tokio::test]
    async fn writer_create_write_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("writer_test.bin");
        let data = b"async writer test data";

        let mut writer = TokioWriter::create(&path).await.unwrap();
        writer.write_all(data).await.unwrap();
        writer.sync_all().await.unwrap();

        let read_back = std::fs::read(&path).unwrap();
        assert_eq!(read_back, data);
    }

    #[tokio::test]
    async fn writer_write_at() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("write_at.bin");

        let mut writer = TokioWriter::create(&path).await.unwrap();
        writer.write_all(&[0u8; 20]).await.unwrap();
        writer.write_at(5, b"hello").await.unwrap();
        writer.sync_all().await.unwrap();

        let read_back = std::fs::read(&path).unwrap();
        assert_eq!(&read_back[5..10], b"hello");
    }
}
