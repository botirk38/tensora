//! Framework-neutral bridge: loads tensors and exposes raw metadata + bytes.

mod safetensors;
mod serverlessllm;
mod types;

pub use safetensors::SafeTensorsHandle;
pub use serverlessllm::ServerlessLLMHandle;
pub use types::RawTensorView;
