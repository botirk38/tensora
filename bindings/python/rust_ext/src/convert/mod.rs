//! Framework conversion dispatch.

mod dtype_map;
mod torch;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub use torch::raw_to_torch_tensor;

pub struct TensorData {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Vec<u8>,
}

pub fn convert_tensor(
    py: Python<'_>,
    framework: &str,
    tensor_data: TensorData,
    device: &str,
) -> PyResult<PyObject> {
    match framework {
        "torch" => raw_to_torch_tensor(
            py,
            tensor_data.shape,
            tensor_data.dtype,
            tensor_data.data,
            device,
        ),
        _ => Err(PyValueError::new_err(format!(
            "unsupported framework: {}. Supported: torch",
            framework
        ))),
    }
}
