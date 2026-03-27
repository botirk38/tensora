//! SafeTensors format: handles and functions.

use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tensor_store::formats::safetensors::TensorView as SafetensorTensorView;
use tensor_store::formats::safetensors::{self, Dtype, MmapModel, Model};
use tensor_store::serialize;
use tensor_store::TensorView;

use crate::convert::{convert_tensor, extract_tensor_raw, TensorData};
use crate::errors::map_reader_error;

type TensorTuple = (String, Vec<usize>, Dtype, Vec<u8>);

fn validate_path_exists(path: &Path) -> PyResult<()> {
    if !path.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "path not found: {}",
            path.display()
        )));
    }
    Ok(())
}

// --- SafeTensors Handle ---

/// Python-exposed handle for SafeTensors files.
/// Supports both mmap-backed (file-backed) and owned (memory-backed) variants.
#[pyclass]
pub struct SafeTensorsHandlePy {
    inner: SafeTensorsHandleInner,
}

enum SafeTensorsHandleInner {
    Mmap(MmapModel),
    Owned(Model),
}

#[pymethods]
impl SafeTensorsHandlePy {
    #[new]
    #[pyo3(signature = (path,))]
    fn new(path: PathBuf) -> PyResult<Self> {
        validate_path_exists(&path)?;
        let inner = Python::with_gil(|py| {
            py.allow_threads(|| safetensors::MmapModel::load(&path))
                .map_err(map_reader_error)
        })?;
        Ok(SafeTensorsHandlePy {
            inner: SafeTensorsHandleInner::Mmap(inner),
        })
    }

    fn keys(&self) -> Vec<String> {
        match &self.inner {
            SafeTensorsHandleInner::Mmap(h) => {
                h.tensor_names().into_iter().map(String::from).collect()
            }
            SafeTensorsHandleInner::Owned(h) => {
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
        let view = match &self.inner {
            SafeTensorsHandleInner::Mmap(h) => h.tensor(name).map_err(map_reader_error)?,
            SafeTensorsHandleInner::Owned(h) => h.tensor(name).map_err(map_reader_error)?,
        };
        let tensor = convert_tensor(
            py,
            framework,
            TensorData {
                shape: view.shape(),
                dtype: view.dtype(),
                data: view.data(),
            },
            device,
        )?;
        Ok(tensor)
    }
}

// --- SafeTensors Functions ---

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_safetensors(path: PathBuf) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let owned = Python::with_gil(|py| {
        py.allow_threads(|| {
            #[cfg(target_os = "linux")]
            {
                tokio_uring::start(async move { safetensors::Model::load(&path).await })
            }
            #[cfg(not(target_os = "linux"))]
            {
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    ReaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
                })?;
                rt.block_on(safetensors::Model::load(&path))
            }
        })
        .map_err(map_reader_error)
    })?;
    Python::with_gil(|py| {
        let handle = SafeTensorsHandlePy {
            inner: SafeTensorsHandleInner::Owned(owned),
        };
        let handle_py = Py::new(py, handle)?;
        Ok(handle_py.into_pyobject(py)?.into_any().unbind())
    })
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_safetensors_sync(path: PathBuf) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let inner = Python::with_gil(|py| {
        py.allow_threads(|| safetensors::Model::load_sync(&path))
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
        py.allow_threads(|| safetensors::MmapModel::load(&path))
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
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_safetensors(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let owned: Model = py.allow_threads(|| {
        #[cfg(target_os = "linux")]
        {
            tokio_uring::start(async move { safetensors::Model::load(path_ref).await })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                ReaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;
            rt.block_on(safetensors::Model::load(path_ref))
        }
    }).map_err(map_reader_error)?;

    let keys: Vec<String> = owned.tensor_names().into_iter().map(String::from).collect();
    for k in keys {
        let view = owned.tensor(&k).map_err(map_reader_error)?;
        let tensor = convert_tensor(
            py,
            framework,
            TensorData {
                shape: view.shape(),
                dtype: view.dtype(),
                data: view.data(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_safetensors_sync(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_path_exists(&path)?;
    let builtins = py.import("builtins")?;
    let result = builtins.getattr("dict")?.call0()?;

    let path_ref = path.as_path();
    let handle: Model = py
        .allow_threads(|| safetensors::Model::load_sync(path_ref))
        .map_err(map_reader_error)?;
    for k in handle.tensor_names() {
        let view = handle.tensor(k).map_err(map_reader_error)?;
        let tensor = convert_tensor(
            py,
            framework,
            TensorData {
                shape: view.shape(),
                dtype: view.dtype(),
                data: view.data(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_safetensors_mmap(
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
        .allow_threads(|| safetensors::MmapModel::load(path_ref))
        .map_err(map_reader_error)?;
    for k in handle.tensor_names() {
        let view = handle.tensor(k).map_err(map_reader_error)?;
        let tensor = convert_tensor(
            py,
            framework,
            TensorData {
                shape: view.shape(),
                dtype: view.dtype(),
                data: view.data(),
            },
            device,
        )?;
        result.call_method1("__setitem__", (&k, tensor))?;
    }
    Ok(result.into())
}

// --- Save Functions ---

fn str_to_dtype(s: &str) -> PyResult<Dtype> {
    match s {
        "F64" | "float64" => Ok(Dtype::F64),
        "F32" | "float32" => Ok(Dtype::F32),
        "F16" | "float16" => Ok(Dtype::F16),
        "BF16" | "bfloat16" => Ok(Dtype::BF16),
        "I64" | "int64" => Ok(Dtype::I64),
        "I32" | "int32" => Ok(Dtype::I32),
        "I16" | "int16" => Ok(Dtype::I16),
        "I8" | "int8" => Ok(Dtype::I8),
        "U64" | "uint64" => Ok(Dtype::U64),
        "U32" | "uint32" => Ok(Dtype::U32),
        "U16" | "uint16" => Ok(Dtype::U16),
        "U8" | "uint8" => Ok(Dtype::U8),
        "BOOL" | "bool" => Ok(Dtype::BOOL),
        "C64" | "complex64" => Ok(Dtype::C64),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported dtype: {}",
            s
        ))),
    }
}

fn tensors_to_raw(
    py: Python<'_>,
    tensors: &Bound<'_, PyDict>,
    framework: &str,
) -> PyResult<Vec<TensorTuple>> {
    let mut result = Vec::new();

    for (key, value) in tensors.iter() {
        let name: String = key.extract()?;
        let (shape, dtype_str, data) = extract_tensor_raw(py, framework, &value)?;
        let dtype = str_to_dtype(&dtype_str)?;
        result.push((name, shape, dtype, data));
    }

    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (tensors, path, framework="torch", metadata=None))]
pub fn save_safetensors(
    py: Python<'_>,
    tensors: Bound<'_, PyDict>,
    path: PathBuf,
    framework: &str,
    metadata: Option<HashMap<String, String>>,
) -> PyResult<()> {
    let raw_tensors = tensors_to_raw(py, &tensors, framework)?;

    let views: Vec<(String, SafetensorTensorView<'_>)> = raw_tensors
        .iter()
        .map(|(name, shape, dtype, data)| {
            let view = SafetensorTensorView::new(*dtype, shape.clone(), data.as_slice())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok((name.clone(), view))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let serialized = serialize(views.iter().map(|(n, v)| (n.as_str(), v)), metadata)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    std::fs::write(&path, serialized).map_err(pyo3::exceptions::PyIOError::new_err)?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (tensors, framework="torch", metadata=None))]
pub fn save_safetensors_bytes(
    py: Python<'_>,
    tensors: Bound<'_, PyDict>,
    framework: &str,
    metadata: Option<HashMap<String, String>>,
) -> PyResult<PyObject> {
    let raw_tensors = tensors_to_raw(py, &tensors, framework)?;

    let views: Vec<(String, SafetensorTensorView<'_>)> = raw_tensors
        .iter()
        .map(|(name, shape, dtype, data)| {
            let view = SafetensorTensorView::new(*dtype, shape.clone(), data.as_slice())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok((name.clone(), view))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let serialized = serialize(views.iter().map(|(n, v)| (n.as_str(), v)), metadata)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let bytes_obj = pyo3::types::PyBytes::new(py, &serialized);
    Ok(bytes_obj.into())
}
