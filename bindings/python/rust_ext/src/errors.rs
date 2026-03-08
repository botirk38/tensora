//! Python exception mapping for tensor_store errors.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use tensor_store::ReaderError;

/// Custom Python exception for tensor_store errors.
#[pyclass(extends = PyException)]
#[derive(Debug)]
pub struct TensorStoreError;

/// Register TensorStoreError on the module so Python can catch it.
pub fn register_tensor_store_error(
    py: Python<'_>,
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    let exc = py.get_type::<TensorStoreError>();
    exc.setattr("__module__", module.name()?)?;
    module.add("TensorStoreError", exc)?;
    Ok(())
}

/// Map ReaderError to PyErr using TensorStoreError when available.
pub fn map_reader_error(e: ReaderError) -> PyErr {
    let msg = format!("tensor_store error: {e}");
    PyErr::new::<PyException, _>(msg)
}

/// Map a missing tensor to a Python ValueError.
pub fn tensor_not_found(name: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("tensor not found: {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_map_reader_error() {
        let io_err = ReaderError::Io(io::Error::new(io::ErrorKind::NotFound, "file not found"));
        let py_err = map_reader_error(io_err);
        let msg = format!("{py_err}");
        assert!(msg.contains("tensor_store error"));
        assert!(msg.contains("file not found"));
    }
}
