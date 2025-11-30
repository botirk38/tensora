//! Checkpoint format readers.
//!
//! This module provides functionality to parse and deserialize different
//! checkpoint formats. Each format reader extracts metadata and provides
//! structured access to tensor information.
//!
//! # Architecture
//!
//! ```text
//! readers/
//! ├── safetensors.rs     Parse SafeTensors format
//! ├── tensorstore.rs     Parse TensorStore index
//! ├── serverlessllm.rs   Parse ServerlessLLM index
//! ├── traits.rs          Common reader traits
//! ├── error.rs           Unified error types
//! └── mod.rs
//! ```
//!
//! # Design Philosophy
//!
//! - **Trait-based**: Common `AsyncReader` and `SyncReader` traits for all formats
//! - **Unified errors**: Single `ReaderError` type with format-specific variants
//! - **Streaming**: Parse metadata without loading all data into memory
//! - **Structured**: Return typed data structures, not raw bytes
//! - **Validation**: Check format integrity during parsing
//! - **Extensible**: Easy to add new format readers
//!
//! # Example Usage
//!
//! ## `SafeTensors` Format
//!
//! ```rust,ignore
//! use tensor_store::readers::safetensors;
//! use tensor_store::readers::traits::AsyncReader;
//!
//! // Using free function
//! let tensors = safetensors::load("model.safetensors").await?;
//!
//! // Or using trait
//! use tensor_store::readers::safetensors::OwnedSafeTensors;
//! let tensors = OwnedSafeTensors::load("model.safetensors").await?;
//!
//! // Access tensor information
//! for name in tensors.names() {
//!     let view = tensors.tensor(name)?;
//!     println!("Tensor: {} ({:?})", name, view.shape());
//! }
//! ```
//!
//! ## `ServerlessLLM` Format
//!
//! ```rust,ignore
//! use tensor_store::readers::serverlessllm;
//! use tensor_store::readers::traits::TensorMetadata;
//!
//! // Parse index file
//! let index = serverlessllm::parse_index("tensor_index.json").await?;
//!
//! // Use metadata trait
//! println!("Found {} tensors", index.len());
//! for name in index.tensor_names() {
//!     if let Some(entry) = index.get(name) {
//!         println!("{}: {} bytes in partition {}", name, entry.size, entry.partition_id);
//!     }
//! }
//!
//! // Iterate over tensors
//! for (name, entry) in &index {
//!     println!("{}: offset={}, size={}", name, entry.offset, entry.size);
//! }
//! ```
//!
//! ## Error Handling
//!
//! ```rust,ignore
//! use tensor_store::readers::error::{ReaderError, ReaderResult};
//!
//! async fn load_any_format(path: &str) -> ReaderResult<()> {
//!     match safetensors::load(path).await {
//!         Ok(tensors) => println!("Loaded {} tensors", tensors.len()),
//!         Err(ReaderError::SafeTensors(e)) => eprintln!("SafeTensors error: {}", e),
//!         Err(ReaderError::Io(e)) => eprintln!("I/O error: {}", e),
//!         Err(e) => eprintln!("Other error: {}", e),
//!     }
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod safetensors;
pub mod serverlessllm;
pub mod traits;

// Re-export commonly used types
pub use error::{ReaderError, ReaderResult};
pub use traits::{AsyncReader, SyncReader, TensorMetadata};
