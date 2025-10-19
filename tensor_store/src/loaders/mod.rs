#[cfg(target_os = "linux")]
pub mod uring;

pub mod tokio;

pub use std::io::Result as IoResult;
