//! Traits for tensor model access and checkpoint serialization.
//!
//! Three domain traits, one per type:
//! - [`Tensor`]     — read-only view of a single tensor's bytes, shape, and dtype
//! - [`Model`]      — read-only access to a collection of named tensors
//! - [`Checkpoint`] — loading and saving a format-specific checkpoint

pub mod checkpoint;
pub mod model;
pub mod tensor;

pub use checkpoint::Checkpoint;
pub use model::Model;
pub use tensor::Tensor;
