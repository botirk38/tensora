//! ServerlessLLM format: handles and functions.

use pyo3::prelude::*;
use std::path::PathBuf;

use tensor_store::formats::serverlessllm::{self, ServerlessLLMMmap, ServerlessLLMOwned};
use tensor_store::TensorMetadata;

use crate::convert::{convert_tensor, TensorData};
use crate::errors::{map_reader_error, tensor_not_found};
use super::runtime_async::load_serverlessllm_async;
use super::validation::validate_path_exists;

// --- ServerlessLLM Handle ---

/// Python-exposed handle for ServerlessLLM files.
/// Supports both mmap-backed (file-backed) and owned (memory-backed) variants.
#[pyclass]
pub struct ServerlessLLMHandlePy {
    inner: ServerlessLLMHandleInner,
}

enum ServerlessLLMHandleInner {
    Mmap(ServerlessLLMMmap),
    Owned(ServerlessLLMOwned),
}

#[pymethods]
impl ServerlessLLMHandlePy {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        validate_path_exists(&path)?;
        let inner = Python::with_gil(|py| {
            py.allow_threads(|| serverlessllm::load_mmap(&path))
                .map_err(map_reader_error)
        })?;
        Ok(ServerlessLLMHandlePy {
            inner: ServerlessLLMHandleInner::Mmap(inner),
        })
    }

    fn keys(&self) -> Vec<String> {
        match &self.inner {
            ServerlessLLMHandleInner::Mmap(h) => h.tensor_names().into_iter().map(String::from).collect(),
            ServerlessLLMHandleInner::Owned(h) => h.tensor_names().into_iter().map(String::from).collect(),
        }
    }

    #[pyo3(signature = (name, device="cpu"))]
    fn get_tensor(&self, py: Python<'_>, name: &str, device: &str) -> PyResult<PyObject> {
        let tensor = match &self.inner {
            ServerlessLLMHandleInner::Mmap(h) => {
                let tensor = h.tensor(name).ok_or_else(|| tensor_not_found(name))?;
                convert_tensor(
                    py,
                    "torch",
                    TensorData {
                        shape: tensor.shape().to_vec(),
                        dtype: tensor.dtype().to_string(),
                        data: tensor.data().to_vec(),
                    },
                    device,
                )?
            }
            ServerlessLLMHandleInner::Owned(h) => {
                let tensor = h.tensor(name).ok_or_else(|| tensor_not_found(name))?;
                convert_tensor(
                    py,
                    "torch",
                    TensorData {
                        shape: tensor.shape().to_vec(),
                        dtype: tensor.dtype().to_string(),
                        data: tensor.data().to_vec(),
                    },
                    device,
                )?
            }
        };
        Ok(tensor)
    }
}

// --- ServerlessLLM Functions ---

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_serverlessllm(py: Python<'_>, path: PathBuf) -> PyResult<PyObject> {
    let awaitable = pyo3_async_runtimes::tokio::future_into_py(py, async move {
        validate_path_exists(&path)?;
        let inner = serverlessllm::load_mmap(&path).map_err(map_reader_error)?;
        Python::with_gil(|py| {
            Ok(Py::new(py, ServerlessLLMHandlePy {
                inner: ServerlessLLMHandleInner::Mmap(inner),
            })?
            .into_pyobject(py)?
            .into_any()
            .unbind())
        })
    })?;
    Ok(awaitable.unbind())
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_serverlessllm_sync(path: PathBuf) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let inner = Python::with_gil(|py| {
        py.allow_threads(|| serverlessllm::load_sync(&path))
            .map_err(map_reader_error)
    })?;
    Python::with_gil(|py| {
        let handle = ServerlessLLMHandlePy {
            inner: ServerlessLLMHandleInner::Owned(inner),
        };
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_serverlessllm_mmap(path: PathBuf) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let inner = Python::with_gil(|py| {
        py.allow_threads(|| serverlessllm::load_mmap(&path))
            .map_err(map_reader_error)
    })?;
    Python::with_gil(|py| {
        let handle = ServerlessLLMHandlePy {
            inner: ServerlessLLMHandleInner::Mmap(inner),
        };
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path, device="cpu"))]
pub fn load_serverlessllm(py: Python<'_>, path: PathBuf, device: &str) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let path = path.to_path_buf();
    let device = device.to_string();

    let awaitable = pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let owned: ServerlessLLMOwned = load_serverlessllm_async(path)
            .await
            .map_err(map_reader_error)?;
        
        let tensor_names: Vec<String> = owned.tensor_names().into_iter().map(String::from).collect();

        let tensors_data: Vec<(String, Vec<usize>, String, Vec<u8>)> = tensor_names
            .iter()
            .map(|k| {
                let tensor = owned.tensor(k).ok_or_else(|| tensor_not_found(k))?;
                Ok((
                    k.clone(),
                    tensor.shape().to_vec(),
                    tensor.dtype().to_string(),
                    tensor.data().to_vec(),
                ))
            })
            .collect::<PyResult<_>>()?;

        Python::with_gil(|py| {
            let builtins = py.import("builtins")?;
            let result = builtins.getattr("dict")?.call0()?;
            for (k, shape, dtype, data) in tensors_data {
                let tensor = convert_tensor(
                    py,
                    "torch",
                    TensorData { shape, dtype, data },
                    &device,
                )?;
                result.call_method1("__setitem__", (k, tensor))?;
            }
            Ok(result.unbind())
        })
    })?;
    Ok(awaitable.unbind())
}

#[pyfunction]
#[pyo3(signature = (path, device="cpu"))]
pub fn load_serverlessllm_sync(
    py: Python<'_>,
    path: PathBuf,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let handle: ServerlessLLMOwned = py
        .allow_threads(|| serverlessllm::load_sync(path_ref))
        .map_err(map_reader_error)?;
    
    let tensor_names: Vec<String> = handle.tensor_names().into_iter().map(String::from).collect();
    for k in tensor_names {
        let tensor = handle.tensor(&k).ok_or_else(|| tensor_not_found(&k))?;
        let tensor = convert_tensor(
            py,
            "torch",
            TensorData {
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype().to_string(),
                data: tensor.data().to_vec(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (path, device="cpu"))]
pub fn load_serverlessllm_mmap(
    py: Python<'_>,
    path: PathBuf,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let handle: ServerlessLLMMmap = py
        .allow_threads(|| serverlessllm::load_mmap(path_ref))
        .map_err(map_reader_error)?;
    
    let tensor_names: Vec<String> = handle.tensor_names().into_iter().map(String::from).collect();
    for k in tensor_names {
        let tensor = handle.tensor(&k).ok_or_else(|| tensor_not_found(&k))?;
        let tensor = convert_tensor(
            py,
            "torch",
            TensorData {
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype().to_string(),
                data: tensor.data().to_vec(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}
