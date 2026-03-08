//! Maps tensor_store/safetensors dtypes to PyTorch dtype names.

/// Returns the torch dtype attribute name for a safetensors dtype string.
/// Used when calling torch.<dtype> or getattr(torch, name).
pub fn safetensors_dtype_to_torch_name(dtype: &str) -> Option<&'static str> {
    match dtype {
        "F64" | "float64" | "torch.float64" => Some("float64"),
        "F32" | "float32" | "torch.float32" => Some("float32"),
        "F16" | "float16" | "torch.float16" => Some("float16"),
        "BF16" | "bfloat16" | "torch.bfloat16" => Some("bfloat16"),
        "I64" | "int64" | "torch.int64" => Some("int64"),
        "I32" | "int32" | "torch.int32" => Some("int32"),
        "I16" | "int16" | "torch.int16" => Some("int16"),
        "I8" | "int8" | "torch.int8" => Some("int8"),
        "U64" | "uint64" | "torch.uint64" => Some("uint64"),
        "U32" | "uint32" | "torch.uint32" => Some("uint32"),
        "U16" | "uint16" | "torch.uint16" => Some("uint16"),
        "U8" | "uint8" | "torch.uint8" => Some("uint8"),
        "BOOL" | "bool" | "torch.bool" => Some("bool"),
        "F8_E4M3" | "float8_e4m3fn" | "torch.float8_e4m3fn" => Some("float8_e4m3fn"),
        "F8_E5M2" | "float8_e5m2" | "torch.float8_e5m2" => Some("float8_e5m2"),
        "C64" | "complex64" | "torch.complex64" => Some("complex64"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_dtype_to_torch_name() {
        assert_eq!(safetensors_dtype_to_torch_name("F32"), Some("float32"));
        assert_eq!(safetensors_dtype_to_torch_name("float32"), Some("float32"));
        assert_eq!(safetensors_dtype_to_torch_name("F64"), Some("float64"));
        assert_eq!(safetensors_dtype_to_torch_name("I64"), Some("int64"));
        assert_eq!(safetensors_dtype_to_torch_name("BF16"), Some("bfloat16"));
        assert_eq!(safetensors_dtype_to_torch_name("BOOL"), Some("bool"));
        assert_eq!(safetensors_dtype_to_torch_name("U8"), Some("uint8"));
        assert_eq!(safetensors_dtype_to_torch_name("unknown"), None);
        assert_eq!(safetensors_dtype_to_torch_name(""), None);
        assert_eq!(
            safetensors_dtype_to_torch_name("torch.float32"),
            Some("float32")
        );
        assert_eq!(
            safetensors_dtype_to_torch_name("torch.int64"),
            Some("int64")
        );
    }
}
