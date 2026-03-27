//! ServerlessLLM format: handles and functions.

use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use std::path::{Path, PathBuf};

use tensor_store::formats::serverlessllm::{self, MmapModel, Model};
use tensor_store::ReaderError;

use crate::convert::{convert_tensor, TensorData};
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

fn load_serverlessllm_with_io_uring(path: &Path) -> Result<Model, ReaderError> {
    let index_path = path.join("tensor_index.json");
    let index = serverlessllm::Index::load_sync(&index_path)?;
    let num_partitions = index.partition_ids().len();

    if num_partitions > 1 {
        #[cfg(target_os = "linux")]
        {
            tokio_uring::start(async move {
                serverlessllm::Model::load_parallel(path).await
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                ReaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;
            rt.block_on(async {
                serverlessllm::Model::load_parallel(path).await
            })
        }
    } else {
        #[cfg(target_os = "linux")]
        {
            tokio_uring::start(async move {
                serverlessllm::Model::load(path).await
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                ReaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;
            rt.block_on(async {
                serverlessllm::Model::load(path).await
            })
        }
    }
}

// --- ServerlessLLM Handle ---

/// Python-exposed handle for ServerlessLLM files.
/// Supports both mmap-backed (file-backed) and owned (memory-backed) variants.
#[pyclass]
pub struct ServerlessLLMHandlePy {
    inner: ServerlessLLMHandleInner,
}

enum ServerlessLLMHandleInner {
    Mmap(MmapModel),
    Owned(Model),
}

#[pymethods]
impl ServerlessLLMHandlePy {
    #[new]
    #[pyo3(signature = (path,))]
    fn new(path: PathBuf) -> PyResult<Self> {
        validate_path_exists(&path)?;
        let inner = Python::with_gil(|py| {
            py.allow_threads(|| serverlessllm::MmapModel::load(&path))
                .map_err(map_reader_error)
        })?;
        Ok(ServerlessLLMHandlePy {
            inner: ServerlessLLMHandleInner::Mmap(inner),
        })
    }

    fn keys(&self) -> Vec<String> {
        match &self.inner {
            ServerlessLLMHandleInner::Mmap(h) => {
                h.tensor_names().into_iter().map(String::from).collect()
            }
            ServerlessLLMHandleInner::Owned(h) => {
                h.tensor_names().into_iter().map(String::from).collect()
            }
        }
    }

    #[pyo3(signature = (name, framework="torch", device="cpu"))]
    fn get_tensor(
        &self,
        py: Python<'_>,
        name: &str,
        framework: &str,
        device: &str,
    ) -> PyResult<PyObject> {
        let tensor = match &self.inner {
            ServerlessLLMHandleInner::Mmap(h) => {
                let tensor = h.tensor(name).ok_or_else(|| tensor_not_found(name))?;
                convert_tensor(
                    py,
                    framework,
                    TensorData {
                        shape: tensor.shape(),
                        dtype: tensor.dtype(),
                        data: tensor.data(),
                    },
                    device,
                )?
            }
            ServerlessLLMHandleInner::Owned(h) => {
                let tensor = h.tensor(name).ok_or_else(|| tensor_not_found(name))?;
                convert_tensor(
                    py,
                    framework,
                    TensorData {
                        shape: tensor.shape(),
                        dtype: tensor.dtype(),
                        data: tensor.data(),
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
pub fn open_serverlessllm(path: PathBuf) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let owned = Python::with_gil(|py| {
        py.allow_threads(|| {
            load_serverlessllm_with_io_uring(&path)
        })
        .map_err(map_reader_error)
    })?;
    Python::with_gil(|py| {
        let handle = ServerlessLLMHandlePy {
            inner: ServerlessLLMHandleInner::Owned(owned),
        };
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_serverlessllm_sync(path: PathBuf) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let inner = Python::with_gil(|py| {
        py.allow_threads(|| serverlessllm::Model::load_sync(&path))
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
        py.allow_threads(|| serverlessllm::MmapModel::load(&path))
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
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_serverlessllm(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let owned: Model = py.allow_threads(|| {
        load_serverlessllm_with_io_uring(&path)
    }).map_err(map_reader_error)?;

    let tensor_names: Vec<String> =
        owned.tensor_names().into_iter().map(String::from).collect();
    for k in tensor_names {
        let tensor = owned.tensor(&k).ok_or_else(|| tensor_not_found(&k))?;
        let tensor = convert_tensor(
            py,
            framework,
            TensorData {
                shape: tensor.shape(),
                dtype: tensor.dtype(),
                data: tensor.data(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
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
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    // Check partition count to decide between parallel and sequential loading
    let index_path = path.join("tensor_index.json");
    let index = serverlessllm::Index::load_sync(&index_path).map_err(map_reader_error)?;
    let num_partitions = index.partition_ids().len();

    let handle: Model = if num_partitions > 1 {
        py.allow_threads(|| serverlessllm::Model::load_parallel_sync(path.as_path()))
            .map_err(map_reader_error)?
    } else {
        py.allow_threads(|| serverlessllm::Model::load_sync(path.as_path()))
            .map_err(map_reader_error)?
    };

    let tensor_names: Vec<String> = handle
        .tensor_names()
        .into_iter()
        .map(String::from)
        .collect();
    for k in tensor_names {
        let tensor = handle.tensor(&k).ok_or_else(|| tensor_not_found(&k))?;
        let tensor = convert_tensor(
            py,
            framework,
            TensorData {
                shape: tensor.shape(),
                dtype: tensor.dtype(),
                data: tensor.data(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}

/// Convert a safetensors file to ServerlessLLM format.
#[pyfunction]
#[pyo3(signature = (input_path, output_dir, partition_count))]
pub fn convert_safetensors_to_serverlessllm(
    input_path: &str,
    output_dir: &str,
    partition_count: usize,
) -> PyResult<()> {
    use tensor_store::converters::safetensors_to_serverlessllm::convert_safetensors_to_serverlessllm as convert_fn;
    #[cfg(target_os = "linux")]
    {
        tokio_uring::start(async {
            convert_fn(input_path, output_dir, partition_count)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("conversion failed: {e}"))
                })
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("runtime failed: {e}")))?;
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
        })?;
        rt.block_on(async {
            convert_fn(input_path, output_dir, partition_count)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("conversion failed: {e}"))
                })
        })
    }
}

#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_serverlessllm_mmap(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let handle: MmapModel = py
        .allow_threads(|| serverlessllm::MmapModel::load(path_ref))
        .map_err(map_reader_error)?;

    let tensor_names: Vec<String> = handle
        .tensor_names()
        .into_iter()
        .map(String::from)
        .collect();
    for k in tensor_names {
        let tensor = handle.tensor(&k).ok_or_else(|| tensor_not_found(&k))?;
        let tensor = convert_tensor(
            py,
            framework,
            TensorData {
                shape: tensor.shape(),
                dtype: tensor.dtype(),
                data: tensor.data(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}
