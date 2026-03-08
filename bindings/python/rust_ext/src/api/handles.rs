//! Python-exposed handle classes for lazy-loaded tensor files.

use pyo3::prelude::*;
use std::path::PathBuf;

use crate::bridge::{SafeTensorsHandle, ServerlessLLMHandle};
use crate::errors::map_reader_error;
use crate::torch::raw_to_torch_tensor;
use super::runtime_async::{load_safetensors_async, load_serverlessllm_async};
use super::validation::validate_path_exists;

/// Backend for sync loading: mmap (zero-copy) or sync (read into memory).
#[derive(Clone, Copy)]
enum SyncLoadBackend {
    Mmap,
    Sync,
}

// --- SafeTensors ---

#[pyclass]
pub struct SafeTensorsHandlePy {
    inner: SafeTensorsHandle,
}

fn open_safetensors_sync(path: PathBuf, backend: SyncLoadBackend) -> PyResult<SafeTensorsHandlePy> {
    validate_path_exists(&path)?;
    let inner = Python::with_gil(|py| {
        py.allow_threads(|| match backend {
            SyncLoadBackend::Mmap => SafeTensorsHandle::open_mmap(&path),
            SyncLoadBackend::Sync => SafeTensorsHandle::open_sync(&path),
        })
        .map_err(map_reader_error)
    })?;
    Ok(SafeTensorsHandlePy { inner })
}

pub fn open_safetensors_mmap_handle(path: PathBuf) -> PyResult<SafeTensorsHandlePy> {
    open_safetensors_sync(path, SyncLoadBackend::Mmap)
}

pub fn open_safetensors_sync_handle(path: PathBuf) -> PyResult<SafeTensorsHandlePy> {
    open_safetensors_sync(path, SyncLoadBackend::Sync)
}

pub async fn open_safetensors_async_handle(path: PathBuf) -> PyResult<SafeTensorsHandlePy> {
    validate_path_exists(&path)?;
    let owned = load_safetensors_async(path).await.map_err(map_reader_error)?;
    let inner = SafeTensorsHandle::from_async(owned);
    Ok(SafeTensorsHandlePy { inner })
}

#[pymethods]
impl SafeTensorsHandlePy {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        open_safetensors_mmap_handle(path)
    }

    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }

    #[pyo3(signature = (name, device="cpu"))]
    fn get_tensor(&self, py: Python<'_>, name: &str, device: &str) -> PyResult<PyObject> {
        let view = self.inner.get_tensor_raw(name).map_err(map_reader_error)?;
        let tensor = raw_to_torch_tensor(py, &view, device)?;
        Ok(tensor)
    }
}

// --- ServerlessLLM ---

#[pyclass]
pub struct ServerlessLLMHandlePy {
    inner: ServerlessLLMHandle,
}

fn open_serverlessllm_sync(path: PathBuf, backend: SyncLoadBackend) -> PyResult<ServerlessLLMHandlePy> {
    validate_path_exists(&path)?;
    let inner = Python::with_gil(|py| {
        py.allow_threads(|| match backend {
            SyncLoadBackend::Mmap => ServerlessLLMHandle::open_mmap(&path),
            SyncLoadBackend::Sync => ServerlessLLMHandle::open_sync(&path),
        })
        .map_err(map_reader_error)
    })?;
    Ok(ServerlessLLMHandlePy { inner })
}

pub fn open_serverlessllm_mmap_handle(path: PathBuf) -> PyResult<ServerlessLLMHandlePy> {
    open_serverlessllm_sync(path, SyncLoadBackend::Mmap)
}

pub fn open_serverlessllm_sync_handle(path: PathBuf) -> PyResult<ServerlessLLMHandlePy> {
    open_serverlessllm_sync(path, SyncLoadBackend::Sync)
}

pub async fn open_serverlessllm_async_handle(path: PathBuf) -> PyResult<ServerlessLLMHandlePy> {
    validate_path_exists(&path)?;
    let owned = load_serverlessllm_async(path).await.map_err(map_reader_error)?;
    let inner = ServerlessLLMHandle::from_async(owned);
    Ok(ServerlessLLMHandlePy { inner })
}

#[pymethods]
impl ServerlessLLMHandlePy {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        open_serverlessllm_mmap_handle(path)
    }

    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }

    #[pyo3(signature = (name, device="cpu"))]
    fn get_tensor(&self, py: Python<'_>, name: &str, device: &str) -> PyResult<PyObject> {
        let view = self.inner.get_tensor_raw(name).map_err(map_reader_error)?;
        let tensor = raw_to_torch_tensor(py, &view, device)?;
        Ok(tensor)
    }
}

// Handle tests require Python and are run via pytest (test_init.py).
