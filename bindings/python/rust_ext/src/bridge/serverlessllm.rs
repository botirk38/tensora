//! ServerlessLLM format bridge.

use std::path::Path;
use tensor_store::serverlessllm::{self, ServerlessLLMMmap, ServerlessLLMOwned};
use tensor_store::types::traits::TensorMetadata;
use tensor_store::ReaderError;

use super::types::RawTensorView;

/// ServerlessLLM backend: mmap (lazy), sync (eager), or async (eager).
pub enum ServerlessLLMBackend {
    Mmap(ServerlessLLMMmap),
    Sync(ServerlessLLMOwned),
    Async(ServerlessLLMOwned),
}

impl ServerlessLLMBackend {
    pub fn keys(&self) -> Vec<String> {
        match self {
            Self::Mmap(m) => m.tensor_names().into_iter().map(String::from).collect(),
            Self::Sync(s) => s.tensor_names().into_iter().map(String::from).collect(),
            Self::Async(s) => s.tensor_names().into_iter().map(String::from).collect(),
        }
    }

    pub fn get_tensor_raw(&self, name: &str) -> Result<RawTensorView, ReaderError> {
        match self {
            Self::Mmap(m) => {
                let t = m.tensor(name).ok_or_else(|| {
                    ReaderError::InvalidMetadata(format!("tensor not found: {name}"))
                })?;
                Ok(RawTensorView {
                    shape: t.shape().to_vec(),
                    dtype: t.dtype().to_string(),
                    data: t.data().to_vec(),
                })
            }
            Self::Sync(s) => {
                let t = s.tensor(name).ok_or_else(|| {
                    ReaderError::InvalidMetadata(format!("tensor not found: {name}"))
                })?;
                Ok(RawTensorView {
                    shape: t.shape().to_vec(),
                    dtype: t.dtype().to_string(),
                    data: t.data().to_vec(),
                })
            }
            Self::Async(s) => {
                let t = s.tensor(name).ok_or_else(|| {
                    ReaderError::InvalidMetadata(format!("tensor not found: {name}"))
                })?;
                Ok(RawTensorView {
                    shape: t.shape().to_vec(),
                    dtype: t.dtype().to_string(),
                    data: t.data().to_vec(),
                })
            }
        }
    }
}

/// ServerlessLLM handle (unified backend).
pub struct ServerlessLLMHandle {
    inner: ServerlessLLMBackend,
}

impl ServerlessLLMHandle {
    pub fn open_mmap(path: &Path) -> Result<Self, ReaderError> {
        let inner = serverlessllm::load_mmap(path)?;
        Ok(Self {
            inner: ServerlessLLMBackend::Mmap(inner),
        })
    }

    pub fn open_sync(path: &Path) -> Result<Self, ReaderError> {
        let inner = serverlessllm::load_sync(path)?;
        Ok(Self {
            inner: ServerlessLLMBackend::Sync(inner),
        })
    }

    /// Create handle from async-loaded owned data.
    pub fn from_async(owned: ServerlessLLMOwned) -> Self {
        Self {
            inner: ServerlessLLMBackend::Async(owned),
        }
    }

    pub fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }

    pub fn get_tensor_raw(&self, name: &str) -> Result<RawTensorView, ReaderError> {
        self.inner.get_tensor_raw(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_serverlessllm_fixture(dir: &std::path::Path) {
        let index = r#"{"x": [0, 24, [2, 3], [3, 1], "torch.float32", 0]}"#;
        std::fs::write(dir.join("tensor_index.json"), index).unwrap();
        let data = vec![0u8; 24]; // F32 [2,3] = 24 bytes
        std::fs::write(dir.join("tensor.data_0"), data).unwrap();
    }

    #[test]
    fn test_open_mmap() {
        let dir = tempfile::TempDir::new().unwrap();
        write_serverlessllm_fixture(dir.path());

        let handle = ServerlessLLMHandle::open_mmap(dir.path()).unwrap();
        assert_eq!(handle.keys(), vec!["x"]);
        let view = handle.get_tensor_raw("x").unwrap();
        assert_eq!(view.shape, vec![2, 3]);
        assert!(view.dtype.contains("float32") || view.dtype.contains("torch.float32"));
    }

    #[test]
    fn test_open_sync() {
        let dir = tempfile::TempDir::new().unwrap();
        write_serverlessllm_fixture(dir.path());

        let handle = ServerlessLLMHandle::open_sync(dir.path()).unwrap();
        assert_eq!(handle.keys(), vec!["x"]);
        let view = handle.get_tensor_raw("x").unwrap();
        assert_eq!(view.shape, vec![2, 3]);
    }
}
