//! SafeTensors format: handles and functions.

use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tensor_store::formats::safetensors::TensorView as SafetensorTensorView;
use tensor_store::formats::safetensors::{Dtype, MmapModel, Model};
use tensor_store::serialize;
use tensor_store::TensorView;

use super::run_async;
use crate::convert::{
    convert_tensor, convert_tensor_with_context, extract_tensor_raw, TensorData, TorchContext,
};
use crate::errors::map_reader_error;

type TensorTuple = (String, Vec<usize>, Dtype, Vec<u8>);

fn validate_directory(path: &Path) -> PyResult<()> {
    if !path.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "path not found: {}",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(PyValueError::new_err(format!(
            "expected a directory of .safetensors shards: {}",
            path.display()
        )));
    }
    Ok(())
}

#[pyclass]
pub struct SafeTensorsHandlePy {
    inner: MmapModel,
}

#[pymethods]
impl SafeTensorsHandlePy {
    #[new]
    #[pyo3(signature = (path,))]
    fn new(path: PathBuf) -> PyResult<Self> {
        validate_directory(&path)?;
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
        let view = self.inner.tensor(name).map_err(map_reader_error)?;
        convert_tensor(
            py,
            framework,
            TensorData {
                shape: view.shape(),
                dtype: view.dtype(),
                data: view.data(),
            },
            device,
        )
    }
}

fn load_dict_from_model(
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
        let view = model.tensor(name).map_err(map_reader_error)?;
        let tensor = convert_tensor_with_context(
            py,
            framework,
            TensorData {
                shape: view.shape(),
                dtype: view.dtype(),
                data: view.data(),
            },
            device,
            torch_context.as_mut(),
        )?;
        dict.set_item(name.as_ref(), tensor)?;
    }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub fn open_safetensors(path: PathBuf) -> PyResult<PyObject> {
    validate_directory(&path)?;
    Python::with_gil(|py| {
        let handle = SafeTensorsHandlePy {
            inner: py
                .allow_threads(|| MmapModel::open(&path))
                .map_err(map_reader_error)?,
        };
        Ok(Py::new(py, handle)?.into_pyobject(py)?.into_any().unbind())
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
    validate_directory(&path)?;
    let model = py
        .allow_threads(|| run_async(Model::load(path)))
        .map_err(map_reader_error)?;
    load_dict_from_model(py, &model, framework, device)
}

#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_safetensors_async(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_directory(&path)?;
    let model = py
        .allow_threads(|| run_async(Model::load_async(path)))
        .map_err(map_reader_error)?;
    load_dict_from_model(py, &model, framework, device)
}

#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_safetensors_sync(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_directory(&path)?;
    let model = py
        .allow_threads(|| Model::load_sync(path))
        .map_err(map_reader_error)?;
    load_dict_from_model(py, &model, framework, device)
}

#[cfg(target_os = "linux")]
#[pyfunction]
#[pyo3(signature = (path, framework="torch", device="cpu"))]
pub fn load_safetensors_io_uring(
    py: Python<'_>,
    path: PathBuf,
    framework: &str,
    device: &str,
) -> PyResult<PyObject> {
    validate_directory(&path)?;
    let model = py
        .allow_threads(|| Model::load_io_uring(path))
        .map_err(map_reader_error)?;
    load_dict_from_model(py, &model, framework, device)
}

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
        _ => Err(PyValueError::new_err(format!("unsupported dtype: {s}"))),
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
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok((name.clone(), view))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let serialized = serialize(views.iter().map(|(n, v)| (n.as_str(), v)), metadata)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok((name.clone(), view))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let serialized = serialize(views.iter().map(|(n, v)| (n.as_str(), v)), metadata)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pyo3::types::PyBytes::new(py, &serialized).into())
}
