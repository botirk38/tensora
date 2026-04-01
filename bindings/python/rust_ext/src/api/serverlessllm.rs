//! ServerlessLLM format: handles and functions.

use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::{Path, PathBuf};

use tensor_store::formats::serverlessllm::{MmapModel, Model};

use super::run_async;
use crate::convert::{convert_tensor, convert_tensor_with_context, TensorData, TorchContext};
use crate::errors::{map_reader_error, tensor_not_found};

fn validate_path_exists(path: &Path) -> PyResult<()> {
    if !path.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "path not found: {}",
            path.display()
        )));
    }
    Ok(())
}

#[pyclass]
pub struct ServerlessLLMHandlePy {
    inner: MmapModel,
}

#[pymethods]
impl ServerlessLLMHandlePy {
    #[new]
    #[pyo3(signature = (path,))]
    fn new(path: PathBuf) -> PyResult<Self> {
        validate_path_exists(&path)?;
        let inner = Python::with_gil(|py| py.allow_threads(|| MmapModel::open(&path)))
            .map_err(map_reader_error)?;
        Ok(Self { inner })
    }

    fn keys(&self) -> Vec<String> {
        self.inner
            .tensor_names()
            .iter()
            .map(|name| name.as_ref().to_string())
            .collect()
    }

    #[pyo3(signature = (name, framework="torch", device="cpu"))]
    fn get_tensor(
        &self,
        py: Python<'_>,
        name: &str,
        framework: &str,
        device: &str,
    ) -> PyResult<PyObject> {
        let tensor = self
            .inner
            .tensor(name)
            .ok_or_else(|| tensor_not_found(name))?;
        convert_tensor(
            py,
            framework,
            TensorData {
                shape: tensor.shape(),
                dtype: tensor.dtype(),
                data: tensor.data(),
            },
            device,
        )
    }
}

fn load_into_dict(
    py: Python<'_>,
    model: &Model,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    let mut torch_context = if framework == "torch" {
        Some(TorchContext::new(py)?)
    } else {
        None
    };
    for name in model.tensor_names() {
        let tensor = model
            .tensor(name)
            .ok_or_else(|| tensor_not_found(name.as_ref()))?;
        let value = convert_tensor_with_context(
            py,
            framework,
            TensorData {
                shape: tensor.shape(),
                dtype: tensor.dtype(),
                data: tensor.data(),
            },
            device,
            torch_context.as_mut(),
        )?;
        dict.set_item(name.as_ref(), value)?;
    }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_serverlessllm(path: PathBuf) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    Python::with_gil(|py| {
        let handle = ServerlessLLMHandlePy {
            inner: py
                .allow_threads(|| MmapModel::open(&path))
                .map_err(map_reader_error)?,
        };
        Ok(Py::new(py, handle)?.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_serverlessllm(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let model = py
        .allow_threads(|| run_async(Model::load(path)))
        .map_err(map_reader_error)?;
    load_into_dict(py, &model, framework, device)
}

#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_serverlessllm_async(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let model = py
        .allow_threads(|| run_async(Model::load_async(path)))
        .map_err(map_reader_error)?;
    load_into_dict(py, &model, framework, device)
}

#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_serverlessllm_sync(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let model = py
        .allow_threads(|| Model::load_sync(path))
        .map_err(map_reader_error)?;
    load_into_dict(py, &model, framework, device)
}

#[cfg(target_os = "linux")]
#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_serverlessllm_io_uring(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let model = py
        .allow_threads(|| Model::load_io_uring(path))
        .map_err(map_reader_error)?;
    load_into_dict(py, &model, framework, device)
}
