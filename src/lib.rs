pub mod converters;
pub mod formats;
pub mod hf_model;

// ============================================================================
// Convenience re-exports for common types and functions
// ============================================================================

// Error types
pub use formats::error::{LoadError, LoadResult};
pub use formats::error::{SaveError, SaveResult};

// Traits and backend enums
pub use formats::traits::{Checkpoint, Model, Tensor};
pub use formats::{AsyncBackend, Backend};

// Shared tensor primitives
pub use formats::tensor::{Dtype, TensorMeta};

// SafeTensors types
pub use formats::safetensors::Checkpoint as SafeTensorsCheckpoint;
pub use formats::safetensors::Model as SafeTensorsModel;
pub use formats::safetensors::Tensor as SafeTensorsTensor;
pub use formats::safetensors::TensorEntry as SafeTensorsTensorEntry;

// ServerlessLLM types
pub use formats::serverlessllm::{
    Checkpoint as ServerlessLLMCheckpoint, Index, Model as ServerlessLLMModel, PartitionSizing,
    Tensor as ServerlessLLMTensor, TensorEntry as ServerlessLLMTensorEntry,
};

// Conversion functions
pub use converters::safetensors_to_serverlessllm::{
    ConversionEnginePreference, ConversionPlan, ConversionStats, CopyOp,
    SafeTensorsToServerlessLLM, TensorSource,
};

#[cfg(test)]
pub(crate) mod test_utils {
    /// Run an async block using Tokio on all platforms.
    pub fn run_async<F>(f: F) -> F::Output
    where
        F: std::future::Future,
    {
        tokio::runtime::Runtime::new()
            .expect("tokio runtime creation failed")
            .block_on(f)
    }
}
