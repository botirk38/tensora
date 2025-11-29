use std::error::Error;
use std::fmt;

/// Configuration for demo runs.
#[derive(Debug, Clone)]
pub struct DemoConfig {
    /// Optional fixture name to filter (matches entries under `fixtures/`).
    pub fixture: Option<String>,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self { fixture: None }
    }
}

/// Shared result alias for demo entry points.
pub type DemoResult = Result<(), Box<dyn Error>>;

/// Lightweight error type for user-facing demo issues.
#[derive(Debug)]
pub struct DemoError(pub String);

impl DemoError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for DemoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for DemoError {}

/// Format bytes into human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    let bytes_f64 = f64::from(u32::try_from(bytes).unwrap_or(u32::MAX));
    if bytes >= 1_000_000_000 {
        let gb = if bytes > u64::from(u32::MAX) {
            // For very large files, convert directly
            bytes as f64 / 1_000_000_000.0
        } else {
            bytes_f64 / 1_000_000_000.0
        };
        format!("{:.1} GB", gb)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes_f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes_f64 / 1_000.0)
    } else {
        format!("{} bytes", bytes)
    }
}
