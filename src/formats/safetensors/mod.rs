//! SafeTensors format implementation.
//!
//! This module provides reading and writing support for the SafeTensors format.

pub mod ids;
pub mod model;
pub mod serializer;

pub use ids::{ShardCount, ShardId};
pub use safetensors::serialize;

// Re-export model types
pub use model::{Dtype, MmapModel, Model, SafeTensorError, Tensor};

// Re-export serializer types
pub use serializer::{MetadataMap, TensorView, View, Writer};
