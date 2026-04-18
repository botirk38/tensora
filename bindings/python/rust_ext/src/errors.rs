//! Python exception mapping for tensora errors.

use pyo3::prelude::*;
use tensora::{ReaderError, WriterError};

pub fn map_reader_error(e: ReaderError) -> PyErr {
    let msg = format!("tensora error: {e}");
    pyo3::exceptions::PyValueError::new_err(msg)
}

pub fn tensor_not_found(name: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("tensor not found: {name}"))
}

pub fn map_writer_error(e: WriterError) -> PyErr {
    let msg = format!("tensora error: {e}");
    pyo3::exceptions::PyValueError::new_err(msg)
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
        assert!(msg.contains("tensora error"));
        assert!(msg.contains("file not found"));
    }
}
