#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod backends;
pub mod converters;
pub mod readers;
pub mod types;
pub mod writers;

// ============================================================================
// Convenience re-exports for common types and functions
// ============================================================================

// Error types
pub use readers::error::{ReaderError, ReaderResult};
pub use writers::error::{WriterError, WriterResult};

// Traits
pub use readers::traits::{AsyncReader, SyncReader, TensorMetadata};
pub use writers::traits::{AsyncWriter, SyncWriter};

// Reader types
pub use readers::safetensors::{SafeTensors, SafeTensorsMmap};
pub use readers::serverlessllm::{
    ServerlessLLM, ServerlessLLMIndex, ServerlessLLMMmap, Tensor, TensorMmap,
};

// Writer types
pub use writers::safetensors::SafeTensorsWriter;
pub use writers::serverlessllm::ServerlessLlmWriter;

// Conversion functions
pub use converters::safetensors_to_serverlessllm::convert_safetensors_to_serverlessllm;
