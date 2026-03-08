//! Runtime adapter for running tensor_store async loaders.
//!
//! tensor_store async backends (io_uring on Linux, tokio elsewhere) produce !Send futures.
//! We run them in spawn_blocking so our outer future is Send and works with future_into_py.

use std::path::PathBuf;

/// Run SafeTensors async load in a blocking thread with platform-appropriate runtime.
pub async fn load_safetensors_async(
    path: PathBuf,
) -> Result<tensor_store::safetensors::SafeTensorsOwned, tensor_store::ReaderError> {
    tokio::task::spawn_blocking(move || {
        #[cfg(target_os = "linux")]
        {
            tokio_uring::start(async move { tensor_store::safetensors::load(&path).await })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                tensor_store::ReaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;
            rt.block_on(tensor_store::safetensors::load(&path))
        }
    })
    .await
    .map_err(|e| tensor_store::ReaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
}

/// Run ServerlessLLM async load in a blocking thread with platform-appropriate runtime.
pub async fn load_serverlessllm_async(
    path: PathBuf,
) -> Result<tensor_store::serverlessllm::ServerlessLLMOwned, tensor_store::ReaderError> {
    tokio::task::spawn_blocking(move || {
        #[cfg(target_os = "linux")]
        {
            tokio_uring::start(async move { tensor_store::serverlessllm::load(&path).await })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                tensor_store::ReaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;
            rt.block_on(tensor_store::serverlessllm::load(&path))
        }
    })
    .await
    .map_err(|e| tensor_store::ReaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_store::safetensors::{Dtype, SafeTensorsWriter, TensorView};

    fn sample_safetensors_bytes() -> Vec<u8> {
        let data = vec![0u8; 24];
        let view = TensorView::new(Dtype::F32, vec![2, 3], &data).expect("create tensor view");
        SafeTensorsWriter::new()
            .write_to_buffer([("x", view)], None)
            .expect("serialize")
    }

    #[tokio::test]
    async fn test_load_safetensors_async() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_safetensors_bytes()).unwrap();

        let owned = load_safetensors_async(path.to_path_buf())
            .await
            .expect("load_safetensors_async");
        assert_eq!(owned.tensor_names(), vec!["x"]);
    }
}
