//! Conversion helpers exposed to Python.

use pyo3::prelude::*;
use std::path::PathBuf;

use crate::errors::map_writer_error;

#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, partition_count))]
pub fn convert_safetensors_to_serverlessllm(
    input_dir: PathBuf,
    output_dir: PathBuf,
    partition_count: usize,
) -> PyResult<()> {
    tensora::convert_safetensors_to_serverlessllm_sync(
        input_dir.to_string_lossy().as_ref(),
        output_dir.to_string_lossy().as_ref(),
        partition_count,
    )
    .map_err(map_writer_error)
}
