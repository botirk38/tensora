//! High-level format conversion orchestration.
//!
//! This module provides convenient functions for converting between
//! different checkpoint formats. Conversions are streaming where possible
//! to minimize memory usage.
//!
//! # Supported Conversions
//!
//! - **`SafeTensors` → `TensorStore`**: Convert to custom sharded format
//! - **`SafeTensors` → `ServerlessLLM`**: Convert to partitioned binary format
//!
//! # Example Usage (Future)
//!
//! ```rust,ignore
//! use tensor_store::writers::converters;
//!
//! // Convert SafeTensors to TensorStore (16 shards)
//! converters::safetensors_to_tensorstore(
//!     "model.safetensors",
//!     "model",  // output prefix
//!     16,       // shard count
//! ).await?;
//!
//! // Convert SafeTensors to ServerlessLLM (8 partitions)
//! converters::safetensors_to_serverlessllm(
//!     "model.safetensors",
//!     "output_dir/",
//!     8,  // partition count
//! ).await?;
//! ```

pub mod safetensors_to_serverlessllm;
pub mod safetensors_to_tensorstore;
