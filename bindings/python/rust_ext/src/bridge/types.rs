//! Shared types for raw tensor views.

/// Raw tensor view: shape, dtype string, and owned data bytes.
pub struct RawTensorView {
    pub shape: Vec<i64>,
    pub dtype: String,
    pub data: Vec<u8>,
}
