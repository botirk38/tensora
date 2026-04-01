pub mod backends;
pub mod converters;
pub mod formats;

// ============================================================================
// Convenience re-exports for common types and functions
// ============================================================================

// Error types
pub use formats::error::{ReaderError, ReaderResult};
pub use formats::error::{WriterError, WriterResult};

// Traits
pub use formats::traits::{AsyncSerializer, Model, SyncSerializer, TensorView};

// SafeTensors types (aliased to avoid conflict with ServerlessLLM)
pub use formats::safetensors::MmapModel as SafeTensorsMmapModel;
pub use formats::safetensors::Model as SafeTensorsModel;
pub use formats::safetensors::Writer as SafeTensorsWriter;
pub use formats::safetensors::serialize;

// ServerlessLLM types
pub use formats::serverlessllm::{
    Index, MmapModel as ServerlessLLMMmapModel, Model as ServerlessLLMModel,
    RECOMMENDED_PARTITION_TARGET_BYTES, Tensor, TensorMmap, recommended_partition_count,
};

// Conversion functions
pub use converters::safetensors_to_serverlessllm::convert_safetensors_to_serverlessllm;
pub use converters::safetensors_to_serverlessllm::convert_safetensors_to_serverlessllm_async;
#[cfg(target_os = "linux")]
pub use converters::safetensors_to_serverlessllm::convert_safetensors_to_serverlessllm_io_uring;
pub use converters::safetensors_to_serverlessllm::convert_safetensors_to_serverlessllm_sync;
pub use converters::safetensors_to_serverlessllm::{
    ConversionPlan, ConversionStats, CopyOp, TensorSource,
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
