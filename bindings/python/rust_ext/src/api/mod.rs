//! Python API surface: handles and functions, organized by format.

mod convert;
mod safetensors;
mod serverlessllm;

use std::future::Future;
use std::sync::OnceLock;

use tokio::runtime::Runtime;

pub use convert::convert_safetensors_to_serverlessllm;
pub use safetensors::{
    iter_safetensors, load_safetensors, open_safetensors, save_safetensors, save_safetensors_bytes,
    SafeTensorsHandlePy,
};
pub use serverlessllm::{
    iter_serverlessllm, load_serverlessllm, open_serverlessllm, ServerlessLLMHandlePy,
};

fn run_async<T>(
    future: impl Future<Output = tensora::ReaderResult<T>>,
) -> tensora::ReaderResult<T> {
    static RUNTIME: OnceLock<Result<Runtime, std::io::Error>> = OnceLock::new();

    let runtime = RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    });

    runtime
        .as_ref()
        .map_err(|e| tensora::ReaderError::Io(std::io::Error::new(e.kind(), e.to_string())))?
        .block_on(future)
}
