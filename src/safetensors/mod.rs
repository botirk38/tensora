//! SafeTensors format implementation.
//!
//! This module provides reading and writing support for the SafeTensors format.

pub mod reader;
pub mod writer;

// Re-export reader types
pub use reader::{
    Dtype, SafeTensorError, SafeTensorsMmap, SafeTensorsOwned, Tensor, load, load_mmap,
    load_parallel, load_range_sync, load_sync,
};

// Re-export writer types
pub use writer::{MetadataMap, SafeTensorsWriter, TensorView, View};
