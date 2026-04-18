//! ServerlessLLM format: handles and functions.

use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::{Path, PathBuf};

use tensora::formats::serverlessllm::{MmapModel, Model};
use tensora::ReaderResult;

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
            .map(|name| name.to_string())
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

#[pyclass(unsendable)]
pub struct ServerlessLLMIterPy {
    model: Model,
    names: Vec<String>,
    index: usize,
    framework: String,
    device: String,
}

#[pymethods]
impl ServerlessLLMIterPy {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(String, PyObject)>> {
        if self.index >= self.names.len() {
            return Ok(None);
        }

        let name = &self.names[self.index];
        self.index += 1;

        let tensor = self
            .model
            .tensor(name)
            .ok_or_else(|| tensor_not_found(name))?;
        let value = convert_tensor(
            py,
            &self.framework,
            TensorData {
                shape: tensor.shape(),
                dtype: tensor.dtype(),
                data: tensor.data(),
            },
            &self.device,
        )?;

        Ok(Some((name.clone(), value)))
    }
}

fn load_model(path: PathBuf, backend: &str) -> ReaderResult<Model> {
    match backend {
        "default" | "async" => run_async(Model::load(path)),
        "sync" => Model::load_sync(path),
        #[cfg(target_os = "linux")]
        "io_uring" => Model::load_io_uring(path),
        _ => unreachable!("validated backend"),
    }
}

fn validate_backend(backend: &str) -> PyResult<()> {
    #[cfg(target_os = "linux")]
    let supported = ["default", "sync", "async", "io_uring"];
    #[cfg(not(target_os = "linux"))]
    let supported = ["default", "sync", "async"];

    if supported.contains(&backend) {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported backend: {backend}"
        )))
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
#[pyo3(signature = (path, backend="default", framework="torch", device="cpu"))]
pub fn load_serverlessllm(
    py: Python<'_>,
    path: PathBuf,
    backend: &str,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    validate_backend(backend)?;
    let model = py
        .allow_threads(|| load_model(path, backend))
        .map_err(map_reader_error)?;
    load_into_dict(py, &model, framework, device)
}

#[pyfunction]
#[pyo3(signature = (path, backend="default", framework="torch", device="cpu"))]
pub fn iter_serverlessllm(
    py: Python<'_>,
    path: PathBuf,
    backend: &str,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    validate_backend(backend)?;
    let model = py
        .allow_threads(|| load_model(path, backend))
        .map_err(map_reader_error)?;

    let names: Vec<String> = model.tensor_names().iter().map(|n| n.to_string()).collect();

    let iter = ServerlessLLMIterPy {
        model,
        names,
        index: 0,
        framework: framework.to_string(),
        device: device.to_string(),
    };

    Python::with_gil(|py| Ok(Py::new(py, iter)?.into_pyobject(py)?.into_any().unbind()))
}
