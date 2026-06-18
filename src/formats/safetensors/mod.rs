//! SafeTensors format implementation.
//!
//! This module provides reading and writing support for the SafeTensors format.

pub mod checkpoint;
pub mod ids;
pub mod model;

pub use checkpoint::{Checkpoint, MetadataMap, TensorWriteData};
pub use ids::{ShardCount, ShardId};
pub use model::{Model, Tensor};
pub use safetensors::serialize;
