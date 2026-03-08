#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

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
pub use formats::traits::{AsyncReader, AsyncWriter, SyncReader, SyncWriter, TensorMetadata, TensorView};

// SafeTensors types
pub use formats::safetensors::{SafeTensorsMmap, SafeTensorsOwned, SafeTensorsWriter};

// ServerlessLLM types
pub use formats::serverlessllm::{
    ServerlessLLMIndex, ServerlessLLMMmap, ServerlessLLMOwned, ServerlessLlmWriter, Tensor,
    TensorEntry, TensorMmap,
};

// Conversion functions
pub use converters::safetensors_to_serverlessllm::convert_safetensors_to_serverlessllm;

#[cfg(test)]
pub(crate) mod test_utils {
    /// Run an async block. Uses io_uring on Linux, tokio on other platforms.
    pub fn run_async<F, O>(f: F) -> O
    where
        F: std::future::Future<Output = O>,
    {
        #[cfg(target_os = "linux")]
        {
            tokio_uring::start(f).unwrap()
        }
        #[cfg(not(target_os = "linux"))]
        {
            tokio::runtime::Runtime::new().unwrap().block_on(f)
        }
    }
}
