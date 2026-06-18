//! Tensor format implementations.
//!
//! Modules:
//! - `safetensors` — SafeTensors format (HuggingFace standard)
//! - `serverlessllm` — ServerlessLLM format (partitioned layout)
//! - `traits` — Common interfaces
//! - `error` — Error types
//! - `tensor` — Shared tensor primitives
//!
//! Use `safetensors::Checkpoint::load(path, backend)` or
//! `serverlessllm::Checkpoint::load(path, backend)` to load checkpoints.
//! Both produce a format-specific `Model` type.

pub mod error;
pub mod safetensors;
pub mod serverlessllm;
pub mod tensor;
pub mod traits;

pub use error::{LoadError, LoadResult, SaveError, SaveResult};
pub use tensor::Dtype;
pub use traits::{Checkpoint, Model, Tensor};

/// Explicit blocking I/O backend for [`Checkpoint::load`][crate::formats::traits::Checkpoint::load].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Synchronous blocking I/O.
    Sync,
    /// Linux io_uring (zero-copy batch I/O).
    #[cfg(target_os = "linux")]
    IoUring,
}

/// Explicit async I/O backend for [`Checkpoint::aload`][crate::formats::traits::Checkpoint::aload].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncBackend {
    /// Tokio-based async I/O.
    Tokio,
}
