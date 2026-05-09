use std::fmt;

/// Public backend names used by capability checks and profiling tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Sync,
    Async,
    Mmap,
    IoUring,
}

impl Backend {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sync => "sync",
            Self::Async => "async",
            Self::Mmap => "mmap",
            Self::IoUring => "io-uring",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a backend cannot be used in the current process/environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendUnavailableReason {
    UnsupportedPlatform,
    PermissionDenied,
    MissingKernelFeature,
    InvalidKernelConfiguration,
    MissingDependency,
    FilesystemUnsupported,
    Other(String),
}

impl BackendUnavailableReason {
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedPlatform => "unsupported-platform",
            Self::PermissionDenied => "permission-denied",
            Self::MissingKernelFeature => "missing-kernel-feature",
            Self::InvalidKernelConfiguration => "invalid-kernel-configuration",
            Self::MissingDependency => "missing-dependency",
            Self::FilesystemUnsupported => "filesystem-unsupported",
            Self::Other(_) => "other",
        }
    }
}

impl fmt::Display for BackendUnavailableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => f.write_str("unsupported platform"),
            Self::PermissionDenied => f.write_str("permission denied"),
            Self::MissingKernelFeature => f.write_str("missing kernel feature"),
            Self::InvalidKernelConfiguration => f.write_str("invalid kernel configuration"),
            Self::MissingDependency => f.write_str("missing dependency"),
            Self::FilesystemUnsupported => f.write_str("filesystem unsupported"),
            Self::Other(message) => f.write_str(message),
        }
    }
}

/// Availability status for one backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendAvailability {
    Available,
    Unavailable {
        reason: BackendUnavailableReason,
        details: String,
    },
}

impl BackendAvailability {
    #[must_use]
    pub const fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }

    #[must_use]
    pub fn unavailable(reason: BackendUnavailableReason, details: impl Into<String>) -> Self {
        Self::Unavailable {
            reason,
            details: details.into(),
        }
    }
}

impl fmt::Display for BackendAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Available => f.write_str("available"),
            Self::Unavailable { reason, details } => {
                write!(f, "unavailable: {reason}")?;
                if !details.is_empty() {
                    write!(f, " ({details})")?;
                }
                Ok(())
            }
        }
    }
}

/// Snapshot of all backend capabilities for deterministic selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub sync: BackendAvailability,
    pub async_io: BackendAvailability,
    pub mmap: BackendAvailability,
    pub io_uring: BackendAvailability,
}

impl BackendCapabilities {
    #[must_use]
    pub const fn new(
        sync: BackendAvailability,
        async_io: BackendAvailability,
        mmap: BackendAvailability,
        io_uring: BackendAvailability,
    ) -> Self {
        Self {
            sync,
            async_io,
            mmap,
            io_uring,
        }
    }

    #[must_use]
    pub fn availability(&self, backend: Backend) -> &BackendAvailability {
        match backend {
            Backend::Sync => &self.sync,
            Backend::Async => &self.async_io,
            Backend::Mmap => &self.mmap,
            Backend::IoUring => &self.io_uring,
        }
    }

    #[must_use]
    pub fn is_available(&self, backend: Backend) -> bool {
        self.availability(backend).is_available()
    }

    #[must_use]
    pub fn iter(&self) -> [(Backend, &BackendAvailability); 4] {
        [
            (Backend::Sync, &self.sync),
            (Backend::Async, &self.async_io),
            (Backend::Mmap, &self.mmap),
            (Backend::IoUring, &self.io_uring),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_reason_and_details() {
        let status = BackendAvailability::unavailable(
            BackendUnavailableReason::PermissionDenied,
            "io_uring_setup returned EPERM",
        );

        assert_eq!(
            status.to_string(),
            "unavailable: permission denied (io_uring_setup returned EPERM)"
        );
    }
}
