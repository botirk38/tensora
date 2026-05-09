//! Tensor format implementations.
//!
//! Modules:
//! - `safetensors` — SafeTensors format (HuggingFace standard)
//! - `serverlessllm` — ServerlessLLM format (partitioned layout)
//! - `traits` — Common interfaces
//! - `error` — Error types
//!
//! Use `safetensors::Model::load(path)` or `serverlessllm::Model::load(path)` to load checkpoints.
//! Both implement the `Model` trait from `traits`.

pub mod error;
pub mod safetensors;
pub mod serverlessllm;
pub mod traits;

pub use error::{ReaderError, ReaderResult, WriterError, WriterResult};
pub use traits::{AsyncSerializer, Model, SyncSerializer, TensorView};
