//! Windows Tokio async storage — delegates to the shared implementation.
pub use super::shared::TokioStorage;

#[cfg(test)]
mod tests {
    use crate::storage::tokio::TokioStorage;
    use crate::storage::{AsyncReadableStorage, AsyncWritableStorage, ByteRange, FileRange};
    use crate::test_utils::run_async;
    use tempfile::TempDir;

    fn write_tmp(dir: &TempDir, name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn read_file() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let path = write_tmp(&dir, "file.bin", &data);
        let result = run_async(TokioStorage::new().read_file(&path)).unwrap();
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[test]
    fn read_range() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..100).collect();
        let path = write_tmp(&dir, "range.bin", &data);
        let result = run_async(
            TokioStorage::new().read_range(&path, ByteRange::from_offset_len(10, 20).unwrap()),
        )
        .unwrap();
        assert_eq!(result.as_ref(), &data[10..30]);
    }

    #[test]
    fn read_ranges() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        let path = write_tmp(&dir, "multi.bin", &data);
        let entries = [
            FileRange::new(&path, ByteRange::from_offset_len(0, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(20, 10).unwrap()),
            FileRange::new(&path, ByteRange::from_offset_len(100, 5).unwrap()),
        ];
        let results = run_async(TokioStorage::new().read_ranges(&entries)).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data(), &data[0..10]);
        assert_eq!(results[1].data(), &data[20..30]);
        assert_eq!(results[2].data(), &data[100..105]);
    }

    #[test]
    fn write_file_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.bin");
        let data = b"hello windows tokio";
        run_async(TokioStorage::new().write_file(&path, data)).unwrap();
        run_async(TokioStorage::new().sync_all(&path)).unwrap();
        let result = run_async(TokioStorage::new().read_file(&path)).unwrap();
        assert_eq!(result.as_ref(), data);
    }
}
