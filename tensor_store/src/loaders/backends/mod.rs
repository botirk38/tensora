#[cfg(target_os = "linux")]
pub mod io_uring;

pub mod async_io;

pub use std::io::Result as IoResult;
