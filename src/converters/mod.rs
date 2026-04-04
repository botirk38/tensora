//! High-level format conversion.
//!
//! Supported conversions:
//! - `SafeTensors` → `ServerlessLLM`: Convert to partitioned format
//!
//! Example:
//!
//! ```rust,ignore
//! use tensor_store::converters::safetensors_to_serverlessllm;
//!
//! // Convert to ServerlessLLM
//! safetensors_to_serverlessllm(
//!     "model.safetensors",
//!     "output_dir/",
//!     8,  // partition count
//! ).await?;
//! ```

pub mod safetensors_to_serverlessllm;