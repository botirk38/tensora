//! Python-exposed functions: open_* and load_*.

use pyo3::prelude::*;
use std::path::PathBuf;

use super::handles::{
    open_safetensors_async_handle, open_safetensors_mmap_handle, open_safetensors_sync_handle,
    open_serverlessllm_async_handle, open_serverlessllm_mmap_handle,
    open_serverlessllm_sync_handle,
};
use super::runtime_async::{load_safetensors_async, load_serverlessllm_async};
use crate::bridge::{SafeTensorsHandle, ServerlessLLMHandle};
use crate::errors::map_reader_error;
use crate::torch::raw_to_torch_tensor;

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_safetensors(py: Python<'_>, path: PathBuf) -> PyResult<PyObject> {
    let awaitable = pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let handle_py = open_safetensors_async_handle(path).await?;
        Python::with_gil(|py| {
            Ok(Py::new(py, handle_py)?
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
    Python::with_gil(|py| {
        let handle = open_safetensors_sync_handle(path)?;
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_safetensors_mmap(path: PathBuf) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let handle = open_safetensors_mmap_handle(path)?;
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_serverlessllm(py: Python<'_>, path: PathBuf) -> PyResult<PyObject> {
    let awaitable = pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let handle_py = open_serverlessllm_async_handle(path).await?;
        Python::with_gil(|py| {
            Ok(Py::new(py, handle_py)?
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
    Python::with_gil(|py| {
        let handle = open_serverlessllm_sync_handle(path)?;
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_serverlessllm_mmap(path: PathBuf) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let handle = open_serverlessllm_mmap_handle(path)?;
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

fn load_safetensors_impl(
    py: Python<'_>,
    path: PathBuf,
    device: &str,
) -> PyResult<PyObject> {
    super::validation::validate_path_exists(&path)?;
    let path_ref = path.as_path();
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let handle = py
        .allow_threads(|| SafeTensorsHandle::open_mmap(path_ref))
        .map_err(map_reader_error)?;
    for k in handle.keys() {
        let view = handle.get_tensor_raw(&k).map_err(map_reader_error)?;
        let tensor = raw_to_torch_tensor(py, &view, device)?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (path, device="cpu"))]
pub fn load_safetensors(py: Python<'_>, path: PathBuf, device: &str) -> PyResult<PyObject> {
    super::validation::validate_path_exists(&path)?;
    let path = path.to_path_buf();
    let device = device.to_string();

    let awaitable = pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let handle = load_safetensors_async(path)
            .await
            .map_err(map_reader_error)?;
        let handle = SafeTensorsHandle::from_async(handle);
        let keys_and_views: Vec<(String, crate::bridge::RawTensorView)> = handle
            .keys()
            .into_iter()
            .map(|k| {
                let view = handle.get_tensor_raw(&k).map_err(map_reader_error)?;
                Ok((k, view))
            })
            .collect::<PyResult<_>>()?;

        Python::with_gil(|py| {
            let builtins = py.import("builtins")?;
            let result = builtins.getattr("dict")?.call0()?;
            for (k, view) in keys_and_views {
                let tensor = raw_to_torch_tensor(py, &view, &device)?;
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
    super::validation::validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let handle = py
        .allow_threads(|| SafeTensorsHandle::open_sync(path_ref))
        .map_err(map_reader_error)?;
    for k in handle.keys() {
        let view = handle.get_tensor_raw(&k).map_err(map_reader_error)?;
        let tensor = raw_to_torch_tensor(py, &view, device)?;
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
    load_safetensors_impl(py, path, device)
}

#[pyfunction]
#[pyo3(signature = (path, device="cpu"))]
pub fn load_serverlessllm(py: Python<'_>, path: PathBuf, device: &str) -> PyResult<PyObject> {
    super::validation::validate_path_exists(&path)?;
    let path = path.to_path_buf();
    let device = device.to_string();

    let awaitable = pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let handle = load_serverlessllm_async(path)
            .await
            .map_err(map_reader_error)?;
        let handle = ServerlessLLMHandle::from_async(handle);
        let keys_and_views: Vec<(String, crate::bridge::RawTensorView)> = handle
            .keys()
            .into_iter()
            .map(|k| {
                let view = handle.get_tensor_raw(&k).map_err(map_reader_error)?;
                Ok((k, view))
            })
            .collect::<PyResult<_>>()?;

        Python::with_gil(|py| {
            let builtins = py.import("builtins")?;
            let result = builtins.getattr("dict")?.call0()?;
            for (k, view) in keys_and_views {
                let tensor = raw_to_torch_tensor(py, &view, &device)?;
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
    super::validation::validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let handle = py
        .allow_threads(|| ServerlessLLMHandle::open_sync(path_ref))
        .map_err(map_reader_error)?;
    for k in handle.keys() {
        let view = handle.get_tensor_raw(&k).map_err(map_reader_error)?;
        let tensor = raw_to_torch_tensor(py, &view, device)?;
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
    super::validation::validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let handle = py
        .allow_threads(|| ServerlessLLMHandle::open_mmap(path_ref))
        .map_err(map_reader_error)?;
    for k in handle.keys() {
        let view = handle.get_tensor_raw(&k).map_err(map_reader_error)?;
        let tensor = raw_to_torch_tensor(py, &view, device)?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}
