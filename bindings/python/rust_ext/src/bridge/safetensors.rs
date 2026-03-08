//! SafeTensors format bridge.

use std::path::Path;
use tensor_store::safetensors::{self, SafeTensorsMmap, SafeTensorsOwned};
use tensor_store::ReaderError;

use super::types::RawTensorView;

fn tensor_to_raw(view: &safetensors::Tensor<'_>) -> RawTensorView {
    let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
    let dtype = view.dtype().to_string();
    let data = view.data().to_vec();
    RawTensorView { shape, dtype, data }
}

/// SafeTensors backend: mmap (lazy), sync (eager), or async (eager).
pub enum SafeTensorsBackend {
    Mmap(SafeTensorsMmap),
    Sync(SafeTensorsOwned),
    Async(SafeTensorsOwned),
}

impl SafeTensorsBackend {
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
                let view = m.tensor(name)?;
                Ok(tensor_to_raw(&view))
            }
            Self::Sync(s) => {
                let view = s.tensor(name)?;
                Ok(tensor_to_raw(&view))
            }
            Self::Async(s) => {
                let view = s.tensor(name)?;
                Ok(tensor_to_raw(&view))
            }
        }
    }
}

/// SafeTensors handle (unified backend).
pub struct SafeTensorsHandle {
    inner: SafeTensorsBackend,
}

impl SafeTensorsHandle {
    pub fn open_mmap(path: &Path) -> Result<Self, ReaderError> {
        let inner = safetensors::load_mmap(path)?;
        Ok(Self {
            inner: SafeTensorsBackend::Mmap(inner),
        })
    }

    pub fn open_sync(path: &Path) -> Result<Self, ReaderError> {
        let inner = safetensors::load_sync(path)?;
        Ok(Self {
            inner: SafeTensorsBackend::Sync(inner),
        })
    }

    /// Create handle from async-loaded owned data.
    pub fn from_async(owned: SafeTensorsOwned) -> Self {
        Self {
            inner: SafeTensorsBackend::Async(owned),
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
    use tensor_store::safetensors::{Dtype, SafeTensorsWriter, TensorView};

    fn sample_safetensors_bytes() -> Vec<u8> {
        let data = vec![0u8; 24]; // F32 [2,3] = 24 bytes
        let view = TensorView::new(Dtype::F32, vec![2, 3], &data).expect("create tensor view");
        SafeTensorsWriter::new()
            .write_to_buffer([("x", view)], None)
            .expect("serialize")
    }

    #[test]
    fn test_open_mmap() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_safetensors_bytes()).unwrap();

        let handle = SafeTensorsHandle::open_mmap(&path).unwrap();
        assert_eq!(handle.keys(), vec!["x"]);
        let view = handle.get_tensor_raw("x").unwrap();
        assert_eq!(view.shape, vec![2, 3]);
        assert!(view.dtype.contains("F32") || view.dtype.contains("float32"));
    }

    #[test]
    fn test_open_sync() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_safetensors_bytes()).unwrap();

        let handle = SafeTensorsHandle::open_sync(&path).unwrap();
        assert_eq!(handle.keys(), vec!["x"]);
        let view = handle.get_tensor_raw("x").unwrap();
        assert_eq!(view.shape, vec![2, 3]);
    }
}
