//! SafeTensors format: handles and functions.

use pyo3::prelude::*;
use std::path::PathBuf;

use tensor_store::formats::safetensors::{self, SafeTensorsMmap, SafeTensorsOwned};
use tensor_store::TensorView;

use crate::convert::{convert_tensor, TensorData};
use crate::errors::map_reader_error;
use super::runtime_async::load_safetensors_async;
use super::validation::validate_path_exists;

// --- SafeTensors Handle ---

/// Python-exposed handle for SafeTensors files.
/// Supports both mmap-backed (file-backed) and owned (memory-backed) variants.
#[pyclass]
pub struct SafeTensorsHandlePy {
    inner: SafeTensorsHandleInner,
}

enum SafeTensorsHandleInner {
    Mmap(SafeTensorsMmap),
    Owned(SafeTensorsOwned),
}

#[pymethods]
impl SafeTensorsHandlePy {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        validate_path_exists(&path)?;
        let inner = Python::with_gil(|py| {
            py.allow_threads(|| safetensors::load_mmap(&path))
                .map_err(map_reader_error)
        })?;
        Ok(SafeTensorsHandlePy {
            inner: SafeTensorsHandleInner::Mmap(inner),
        })
    }

    fn keys(&self) -> Vec<String> {
        match &self.inner {
            SafeTensorsHandleInner::Mmap(h) => h.tensor_names().into_iter().map(String::from).collect(),
            SafeTensorsHandleInner::Owned(h) => h.tensor_names().into_iter().map(String::from).collect(),
        }
    }

    #[pyo3(signature = (name, device="cpu"))]
    fn get_tensor(&self, py: Python<'_>, name: &str, device: &str) -> PyResult<PyObject> {
        let view = match &self.inner {
            SafeTensorsHandleInner::Mmap(h) => h.tensor(name).map_err(map_reader_error)?,
            SafeTensorsHandleInner::Owned(h) => h.tensor(name).map_err(map_reader_error)?,
        };
        let tensor = convert_tensor(
            py,
            "torch",
            TensorData {
                shape: view.shape().to_vec(),
                dtype: view.dtype().to_string(),
                data: view.data().to_vec(),
            },
            device,
        )?;
        Ok(tensor)
    }
}

// --- SafeTensors Functions ---

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_safetensors(py: Python<'_>, path: PathBuf) -> PyResult<PyObject> {
    let awaitable = pyo3_async_runtimes::tokio::future_into_py(py, async move {
        validate_path_exists(&path)?;
        let owned: SafeTensorsOwned = load_safetensors_async(path)
            .await
            .map_err(map_reader_error)?;
        Python::with_gil(|py| {
            Ok(Py::new(py, SafeTensorsHandlePy {
                inner: SafeTensorsHandleInner::Owned(owned),
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
pub fn open_safetensors_sync(path: PathBuf) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let inner = Python::with_gil(|py| {
        py.allow_threads(|| safetensors::load_sync(&path))
            .map_err(map_reader_error)
    })?;
    Python::with_gil(|py| {
        let handle = SafeTensorsHandlePy {
            inner: SafeTensorsHandleInner::Owned(inner),
        };
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_safetensors_mmap(path: PathBuf) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let inner = Python::with_gil(|py| {
        py.allow_threads(|| safetensors::load_mmap(&path))
            .map_err(map_reader_error)
    })?;
    Python::with_gil(|py| {
        let handle = SafeTensorsHandlePy {
            inner: SafeTensorsHandleInner::Mmap(inner),
        };
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path, device="cpu"))]
pub fn load_safetensors(py: Python<'_>, path: PathBuf, device: &str) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let path = path.to_path_buf();
    let device = device.to_string();

    let awaitable = pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let owned: SafeTensorsOwned = load_safetensors_async(path)
            .await
            .map_err(map_reader_error)?;
        let keys: Vec<String> = owned.tensor_names().into_iter().map(String::from).collect();

        let tensors_data: Vec<(String, Vec<usize>, String, Vec<u8>)> = keys
            .iter()
            .map(|k| {
                let view = owned.tensor(k).map_err(map_reader_error)?;
                Ok((
                    k.clone(),
                    view.shape().to_vec(),
                    view.dtype().to_string(),
                    view.data().to_vec(),
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
pub fn load_safetensors_sync(
    py: Python<'_>,
    path: PathBuf,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let handle: SafeTensorsOwned = py
        .allow_threads(|| safetensors::load_sync(path_ref))
        .map_err(map_reader_error)?;
    for k in handle.tensor_names() {
        let view = handle.tensor(k).map_err(map_reader_error)?;
        let tensor = convert_tensor(
            py,
            "torch",
            TensorData {
                shape: view.shape().to_vec(),
                dtype: view.dtype().to_string(),
                data: view.data().to_vec(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (path, device="cpu"))]
pub fn load_safetensors_mmap(
    py: Python<'_>,
    path: PathBuf,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let handle: SafeTensorsMmap = py
        .allow_threads(|| safetensors::load_mmap(path_ref))
        .map_err(map_reader_error)?;
    for k in handle.tensor_names() {
        let view = handle.tensor(k).map_err(map_reader_error)?;
        let tensor = convert_tensor(
            py,
            "torch",
            TensorData {
                shape: view.shape().to_vec(),
                dtype: view.dtype().to_string(),
                data: view.data().to_vec(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}
