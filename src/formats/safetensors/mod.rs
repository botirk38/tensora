//! SafeTensors format implementation.
//!
//! This module provides reading and writing support for the SafeTensors format.

pub mod checkpoint;
pub mod ids;
pub mod model;
pub mod tensor;

pub use checkpoint::{Checkpoint, MetadataMap};
pub use ids::{ShardCount, ShardId};
pub use model::Model;
pub use safetensors::serialize;
pub use tensor::Tensor;
