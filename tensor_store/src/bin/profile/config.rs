use std::error::Error;
use std::fmt;

/// Common configuration for profiling runs.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Number of iterations to run. Defaults to 1 to keep flamegraphs focused.
    pub iterations: usize,
    /// Optional fixture name (matches entries under `fixtures/`).
    pub fixture: Option<String>,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            fixture: None,
        }
    }
}

impl ProfileConfig {
    /// Ensures iterations are never zero.
    pub fn normalized_iterations(&self) -> usize {
        self.iterations.max(1)
    }
}

/// Shared result alias for profiling entry points.
pub type ProfileResult = Result<(), Box<dyn Error>>;

/// Lightweight error type for user-facing configuration issues.
#[derive(Debug)]
pub struct ProfileError(pub String);

impl ProfileError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ProfileError {}
