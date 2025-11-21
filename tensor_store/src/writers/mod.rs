//! Checkpoint writers module for serializing tensor data to various formats.
//!
//! This module provides functionality to write model checkpoints in various formats.
//! It focuses on format-specific serialization logic and delegates I/O operations
//! to the top-level `backends` module.
//!
//! # Architecture
//!
//! ```text
//! writers/
//! ├── safetensors.rs     Wrap SafeTensors serialization
//! ├── serverlessllm.rs   Write ServerlessLLM format
//! ├── tensorstore.rs     Write TensorStore format
//! └── mod.rs
//! ```
//!
//! # Design Philosophy
//!
//! - **Format-specific**: Each writer handles only its target format
//! - **No conversion logic**: Writers expect pre-converted data
//! - **Async-first**: All operations are async for consistency
//! - **Separation of concerns**: Writing vs conversion vs I/O
//!
//! # Example Usage (Future)
//!
//! ```rust,ignore
//! use tensor_store::writers::ServerlessLlmWriter;
//!
//! let writer = ServerlessLlmWriter::new();
//! writer.write_index("tensor_index.json", &tensors).await?;
//! writer.write_partition("tensor.data_0", 0, &data).await?;
//! ```

pub mod safetensors;
pub mod serverlessllm;
pub mod tensorstore;

pub use safetensors::SafeTensorsWriter;
pub use serverlessllm::ServerlessLlmWriter;
pub use std::io::Result as IoResult;
pub use tensorstore::{TensorStoreIndexEntry, TensorStoreWriter};
