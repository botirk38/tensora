//! PyTorch tensor conversion.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::dtype_map::safetensors_dtype_to_torch_name;

/// Builds a torch.Tensor from raw tensor data.
pub fn raw_to_torch_tensor(
    py: Python<'_>,
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
    device: &str,
) -> PyResult<PyObject> {
    let torch = py.import("torch")?;

    let dtype_name = safetensors_dtype_to_torch_name(&dtype).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unsupported dtype for torch: {}", dtype))
    })?;

    let torch_dtype = torch.getattr(dtype_name)?;
    let torch_uint8 = torch.getattr("uint8")?;

    let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    let numel: i64 = shape_i64.iter().product();

    let tensor = if numel == 0 {
        let shape_py: PyObject = shape_i64.into_pyobject(py)?.unbind();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("dtype", torch_dtype)?;
        torch.call_method("zeros", (shape_py,), Some(&kwargs))?
    } else {
        let data_bytes = PyBytes::new(py, &data);
        let buf = data_bytes.as_any();

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("buffer", buf)?;
        kwargs.set_item("dtype", torch_uint8)?;

        let arr = torch.call_method("frombuffer", (), Some(&kwargs))?;
        let view_kwargs = pyo3::types::PyDict::new(py);
        view_kwargs.set_item("dtype", torch_dtype)?;

        let mut tensor = arr.call_method("view", (), Some(&view_kwargs))?;

        let sys = py.import("sys")?;
        let byteorder: String = sys.getattr("byteorder")?.extract()?;
        if byteorder == "big" {
            let numpy_arr = tensor.call_method0("numpy")?;
            let swap_kwargs = pyo3::types::PyDict::new(py);
            swap_kwargs.set_item("inplace", false)?;
            let swapped = numpy_arr.call_method("byteswap", (), Some(&swap_kwargs))?;
            tensor = torch.call_method1("from_numpy", (swapped,))?;
            let view_kwargs = pyo3::types::PyDict::new(py);
            view_kwargs.set_item("dtype", torch.getattr(dtype_name)?)?;
            tensor = tensor.call_method("view", (), Some(&view_kwargs))?;
        }

        let shape_py: PyObject = shape_i64.into_pyobject(py)?.unbind();
        tensor.call_method1("reshape", (shape_py,))?
    };

    let result = if device != "cpu" && device != "cpu:0" {
        let kwargs = pyo3::types::PyDict::new(py);
        tensor.call_method("to", (device,), Some(&kwargs))?
    } else {
        tensor
    };

    Ok(result.into())
}
