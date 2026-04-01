//! tensor_store Python bindings (PyTorch-first, performance-oriented).

mod api;
mod convert;
mod errors;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pyo3::create_exception!(
    _tensor_store_rust,
    TensorStoreError,
    PyException,
    "Error raised by tensor_store Python bindings."
);

use api::{
    convert_safetensors_to_serverlessllm, load_safetensors, load_safetensors_async,
    load_safetensors_sync, load_serverlessllm, load_serverlessllm_async, load_serverlessllm_sync,
    open_safetensors, open_serverlessllm, save_safetensors, save_safetensors_bytes,
    SafeTensorsHandlePy, ServerlessLLMHandlePy,
};
#[cfg(target_os = "linux")]
use api::{load_safetensors_io_uring, load_serverlessllm_io_uring};

/// Python module entry point.
#[pymodule]
#[pyo3(name = "_tensor_store_rust")]
fn _tensor_store_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SafeTensorsHandlePy>()?;
    m.add_class::<ServerlessLLMHandlePy>()?;
    m.add_function(wrap_pyfunction!(open_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(open_serverlessllm, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors_async, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors_sync, m)?)?;
    #[cfg(target_os = "linux")]
    m.add_function(wrap_pyfunction!(load_safetensors_io_uring, m)?)?;
    m.add_function(wrap_pyfunction!(load_serverlessllm, m)?)?;
    m.add_function(wrap_pyfunction!(load_serverlessllm_async, m)?)?;
    m.add_function(wrap_pyfunction!(load_serverlessllm_sync, m)?)?;
    #[cfg(target_os = "linux")]
    m.add_function(wrap_pyfunction!(load_serverlessllm_io_uring, m)?)?;
    m.add_function(wrap_pyfunction!(convert_safetensors_to_serverlessllm, m)?)?;
    m.add_function(wrap_pyfunction!(save_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(save_safetensors_bytes, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TensorStoreError", m.py().get_type::<TensorStoreError>())?;
    Ok(())
}
