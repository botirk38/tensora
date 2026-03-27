//! TensorFlow tensor conversion.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn safetensors_dtype_to_tf_name(dtype: &str) -> Option<&'static str> {
    match dtype {
        "F64" | "float64" => Some("float64"),
        "F32" | "float32" => Some("float32"),
        "F16" | "float16" => Some("float16"),
        "BF16" | "bfloat16" => Some("bfloat16"),
        "I64" | "int64" => Some("int64"),
        "I32" | "int32" => Some("int32"),
        "I16" | "int16" => Some("int16"),
        "I8" | "int8" => Some("int8"),
        "U64" | "uint64" => Some("uint64"),
        "U32" | "uint32" => Some("uint32"),
        "U16" | "uint16" => Some("uint16"),
        "U8" | "uint8" => Some("uint8"),
        "BOOL" | "bool" => Some("bool"),
        "C64" | "complex64" => Some("complex64"),
        _ => None,
    }
}

pub fn raw_to_tensorflow_tensor(
    py: Python<'_>,
    shape: &[usize],
    dtype: &str,
    data: &[u8],
    device: &str,
) -> PyResult<PyObject> {
    let tf = py.import("tensorflow")?;
    let numpy = py.import("numpy")?;

    let dtype_name = safetensors_dtype_to_tf_name(dtype).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported dtype for tensorflow: {}",
            dtype
        ))
    })?;

    let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    let numel: i64 = shape_i64.iter().product();

    let arr = if numel == 0 {
        let shape_py: PyObject = shape_i64.into_pyobject(py)?.unbind();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("shape", shape_py)?;
        kwargs.set_item("dtype", dtype_name)?;
        numpy.call_method("zeros", (), Some(&kwargs))?
    } else {
        let data_bytes = PyBytes::new(py, data);
        let np_uint8 = numpy.getattr("uint8")?;
        let arr = numpy.call_method1("frombuffer", (data_bytes, np_uint8))?;

        let view_kwargs = pyo3::types::PyDict::new(py);
        view_kwargs.set_item("dtype", dtype_name)?;
        let arr = arr.call_method("view", (), Some(&view_kwargs))?;

        let sys = py.import("sys")?;
        let byteorder: String = sys.getattr("byteorder")?.extract()?;
        let arr = if byteorder == "big" {
            let swapped = arr.call_method0("byteswap")?;
            let copy_kwargs = pyo3::types::PyDict::new(py);
            copy_kwargs.set_item("order", "C")?;
            swapped.call_method("astype", (), Some(&copy_kwargs))?
        } else {
            arr
        };

        let shape_py: PyObject = shape_i64.into_pyobject(py)?.unbind();
        arr.call_method1("reshape", (shape_py,))?
    };

    let result = if device != "cpu" && device != "CPU" && device != "/CPU:0" {
        let device_str = if device.starts_with("cuda") || device.starts_with("gpu") {
            if device.contains(':') {
                format!(
                    "/{}:{}",
                    device
                        .replace("cuda", "GPU")
                        .replace("gpu", "GPU")
                        .split(':')
                        .next()
                        .unwrap_or("GPU"),
                    device.split(':').next_back().unwrap_or("0")
                )
            } else {
                format!("/{}:0", device.replace("cuda", "GPU").replace("gpu", "GPU"))
            }
        } else {
            format!("/{}:0", device.to_uppercase())
        };

        let device_py: PyObject = device_str.into_pyobject(py)?.unbind().into();

        let convert_kwargs = pyo3::types::PyDict::new(py);
        convert_kwargs.set_item("dtype", dtype_name)?;
        convert_kwargs.set_item("device", device_py)?;
        let result = tf.call_method("convert_to_tensor", (&arr,), Some(&convert_kwargs))?;
        result
    } else {
        let convert_kwargs = pyo3::types::PyDict::new(py);
        convert_kwargs.set_item("dtype", dtype_name)?;
        tf.call_method("convert_to_tensor", (&arr,), Some(&convert_kwargs))?
    };

    Ok(result.into())
}

pub fn tf_tensor_to_raw(
    _py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
) -> PyResult<(Vec<usize>, String, Vec<u8>)> {
    Python::with_gil(|_py| {
        let numpy_method = tensor.getattr("numpy")?;
        let numpy_arr = numpy_method.call0()?;

        let shape_method = tensor.getattr("shape")?;
        let shape_list_method = shape_method.getattr("as_list")?;
        let shape: Vec<usize> = shape_list_method.call0()?.extract()?;

        let dtype_obj = tensor.getattr("dtype")?;
        let dtype: String = dtype_obj.getattr("name")?.extract()?;

        let contiguous = numpy_arr.call_method0("copy")?;
        let flat = contiguous.call_method1("flatten", ())?;
        let bytes_obj = flat.call_method0("tobytes")?;
        let data: Vec<u8> = bytes_obj.extract()?;

        let dtype_str = match dtype.as_str() {
            "float64" => "F64",
            "float32" => "F32",
            "float16" => "F16",
            "bfloat16" => "BF16",
            "int64" => "I64",
            "int32" => "I32",
            "int16" => "I16",
            "int8" => "I8",
            "uint64" => "U64",
            "uint32" => "U32",
            "uint16" => "U16",
            "uint8" => "U8",
            "bool" => "BOOL",
            "complex64" => "C64",
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported TensorFlow dtype: {}",
                    other
                )));
            }
        };

        Ok((shape, dtype_str.to_string(), data))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(safetensors_dtype_to_tf_name("F32"), Some("float32"));
        assert_eq!(safetensors_dtype_to_tf_name("F64"), Some("float64"));
        assert_eq!(safetensors_dtype_to_tf_name("BF16"), Some("bfloat16"));
        assert_eq!(safetensors_dtype_to_tf_name("I64"), Some("int64"));
        assert_eq!(safetensors_dtype_to_tf_name("BOOL"), Some("bool"));
        assert_eq!(safetensors_dtype_to_tf_name("C64"), Some("complex64"));
        assert_eq!(safetensors_dtype_to_tf_name("unknown"), None);
    }
}
