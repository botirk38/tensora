#[cfg(target_os = "linux")]
pub mod uring;

#[cfg(not(target_os = "linux"))]
pub mod tokio;

pub use std::io::Result as IoResult;
