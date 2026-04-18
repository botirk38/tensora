//! tensora Python bindings (PyTorch-first, performance-oriented).

mod api;
mod convert;
mod errors;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pyo3::create_exception!(
    _tensora_rust,
    TensoraError,
    PyException,
    "Error raised by tensora Python bindings."
);

use api::{
    convert_safetensors_to_serverlessllm, iter_safetensors, iter_serverlessllm, load_safetensors,
    load_serverlessllm, open_safetensors, open_serverlessllm, save_safetensors,
    save_safetensors_bytes, SafeTensorsHandlePy, ServerlessLLMHandlePy,
};

/// Python module entry point.
#[pymodule]
#[pyo3(name = "_tensora_rust")]
fn _tensora_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SafeTensorsHandlePy>()?;
    m.add_class::<ServerlessLLMHandlePy>()?;
    m.add_function(wrap_pyfunction!(open_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(open_serverlessllm, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(iter_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(load_serverlessllm, m)?)?;
    m.add_function(wrap_pyfunction!(iter_serverlessllm, m)?)?;
    m.add_function(wrap_pyfunction!(convert_safetensors_to_serverlessllm, m)?)?;
    m.add_function(wrap_pyfunction!(save_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(save_safetensors_bytes, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TensoraError", m.py().get_type::<TensoraError>())?;
    Ok(())
}
