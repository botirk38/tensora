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
    let converter = tensora::SafeTensorsToServerlessLLM::new(&input_dir, &output_dir, partition_count)
        .map_err(map_writer_error)?;
    converter.convert_sync().map_err(map_writer_error)
}
