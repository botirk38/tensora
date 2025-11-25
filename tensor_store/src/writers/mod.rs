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
//! ├── error.rs          Unified error types for all writers
//! ├── traits.rs         Common traits for async/sync writers
//! ├── safetensors.rs    Wrap SafeTensors serialization
//! ├── serverlessllm.rs  Write ServerlessLLM format
//! ├── tensorstore.rs    Write TensorStore format
//! └── mod.rs
//! ```
//!
//! # Design Philosophy
//!
//! - **Trait-based**: All writers implement `AsyncWriter` and/or `SyncWriter` traits
//! - **Unified error handling**: All writers use `WriterResult` for consistency
//! - **Format-specific**: Each writer handles only its target format
//! - **No conversion logic**: Writers expect pre-converted data
//! - **Sync and Async**: All writers provide both sync and async methods
//! - **Stateless**: Writers are lightweight and can be freely copied
//! - **Separation of concerns**: Writing vs conversion vs I/O
//!
//! # Example Usage
//!
//! ## Async Writing
//!
//! ```rust,ignore
//! use tensor_store::writers::{ServerlessLlmWriter, SafeTensorsWriter};
//!
//! // ServerlessLLM format
//! let writer = ServerlessLlmWriter::new();
//! writer.write_index("tensor_index.json", &tensors).await?;
//! writer.write_partition("tensor.data_0", data).await?;
//!
//! // SafeTensors format
//! let writer = SafeTensorsWriter::new();
//! writer.write_to_file_async(tensors, None, "model.safetensors").await?;
//! ```
//!
//! ## Sync Writing
//!
//! ```rust,ignore
//! use tensor_store::writers::{ServerlessLlmWriter, SafeTensorsWriter};
//!
//! // ServerlessLLM format
//! let writer = ServerlessLlmWriter::new();
//! writer.write_index_sync("tensor_index.json", &tensors)?;
//! writer.write_partition_sync("tensor.data_0", &data)?;
//!
//! // SafeTensors format
//! let writer = SafeTensorsWriter::new();
//! writer.write_to_file(tensors, None, "model.safetensors")?;
//! ```
//!
//! # Error Handling
//!
//! All writers use the unified `WriterResult<T>` type which wraps `WriterError`.
//! This provides consistent error handling across all formats:
//!
//! ```rust,ignore
//! use tensor_store::writers::{WriterError, WriterResult};
//!
//! match writer.write_index("index.json", &tensors).await {
//!     Ok(()) => println!("Success"),
//!     Err(WriterError::InvalidInput(msg)) => eprintln!("Invalid input: {}", msg),
//!     Err(WriterError::Io(e)) => eprintln!("I/O error: {}", e),
//!     Err(e) => eprintln!("Other error: {}", e),
//! }
//! ```

pub mod error;
pub mod safetensors;
pub mod serverlessllm;
pub mod traits;

pub use error::{WriterError, WriterResult};
pub use safetensors::SafeTensorsWriter;
pub use serverlessllm::ServerlessLlmWriter;
pub use traits::{AsyncWriter, SyncWriter};
