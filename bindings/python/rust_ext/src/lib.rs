//! tensor_store Python bindings (PyTorch-first, performance-oriented).

mod api;
mod bridge;
mod errors;
mod torch;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use api::{
    load_safetensors, load_safetensors_mmap, load_safetensors_sync,
    load_serverlessllm, load_serverlessllm_mmap, load_serverlessllm_sync,
    open_safetensors, open_safetensors_mmap, open_safetensors_sync,
    open_serverlessllm, open_serverlessllm_mmap, open_serverlessllm_sync,
    SafeTensorsHandlePy, ServerlessLLMHandlePy,
};

/// Python module entry point.
#[pymodule]
#[pyo3(name = "_tensor_store_rust")]
fn _tensor_store_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SafeTensorsHandlePy>()?;
    m.add_class::<ServerlessLLMHandlePy>()?;
    m.add_function(wrap_pyfunction!(open_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(open_safetensors_sync, m)?)?;
    m.add_function(wrap_pyfunction!(open_safetensors_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(open_serverlessllm, m)?)?;
    m.add_function(wrap_pyfunction!(open_serverlessllm_sync, m)?)?;
    m.add_function(wrap_pyfunction!(open_serverlessllm_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors_sync, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(load_serverlessllm, m)?)?;
    m.add_function(wrap_pyfunction!(load_serverlessllm_sync, m)?)?;
    m.add_function(wrap_pyfunction!(load_serverlessllm_mmap, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    errors::register_tensor_store_error(m.py(), m)?;
    Ok(())
}
