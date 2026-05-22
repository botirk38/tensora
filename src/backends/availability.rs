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

    #[test]
    fn backend_as_str_all_variants() {
        assert_eq!(Backend::Sync.as_str(), "sync");
        assert_eq!(Backend::Async.as_str(), "async");
        assert_eq!(Backend::Mmap.as_str(), "mmap");
        assert_eq!(Backend::IoUring.as_str(), "io-uring");
    }

    #[test]
    fn backend_display_matches_as_str() {
        for backend in [Backend::Sync, Backend::Async, Backend::Mmap, Backend::IoUring] {
            assert_eq!(format!("{backend}"), backend.as_str());
        }
    }

    #[test]
    fn unavailable_reason_code_all_variants() {
        assert_eq!(
            BackendUnavailableReason::UnsupportedPlatform.code(),
            "unsupported-platform"
        );
        assert_eq!(
            BackendUnavailableReason::PermissionDenied.code(),
            "permission-denied"
        );
        assert_eq!(
            BackendUnavailableReason::MissingKernelFeature.code(),
            "missing-kernel-feature"
        );
        assert_eq!(
            BackendUnavailableReason::InvalidKernelConfiguration.code(),
            "invalid-kernel-configuration"
        );
        assert_eq!(
            BackendUnavailableReason::MissingDependency.code(),
            "missing-dependency"
        );
        assert_eq!(
            BackendUnavailableReason::FilesystemUnsupported.code(),
            "filesystem-unsupported"
        );
        assert_eq!(
            BackendUnavailableReason::Other("test".into()).code(),
            "other"
        );
    }

    #[test]
    fn availability_is_available() {
        assert!(BackendAvailability::Available.is_available());
        assert!(!BackendAvailability::unavailable(
            BackendUnavailableReason::PermissionDenied,
            "test"
        )
        .is_available());
    }

    #[test]
    fn display_available() {
        assert_eq!(BackendAvailability::Available.to_string(), "available");
    }

    #[test]
    fn display_unavailable_empty_details() {
        let status = BackendAvailability::unavailable(
            BackendUnavailableReason::UnsupportedPlatform,
            "",
        );
        assert_eq!(status.to_string(), "unavailable: unsupported platform");
    }

    #[test]
    fn display_all_unavailable_reasons() {
        let reasons = [
            (BackendUnavailableReason::UnsupportedPlatform, "unsupported platform"),
            (BackendUnavailableReason::PermissionDenied, "permission denied"),
            (BackendUnavailableReason::MissingKernelFeature, "missing kernel feature"),
            (BackendUnavailableReason::InvalidKernelConfiguration, "invalid kernel configuration"),
            (BackendUnavailableReason::MissingDependency, "missing dependency"),
            (BackendUnavailableReason::FilesystemUnsupported, "filesystem unsupported"),
            (BackendUnavailableReason::Other("custom msg".into()), "custom msg"),
        ];
        for (reason, expected_str) in reasons {
            assert_eq!(format!("{reason}"), expected_str);
        }
    }

    #[test]
    fn capabilities_availability_lookup() {
        let caps = BackendCapabilities::new(
            BackendAvailability::Available,
            BackendAvailability::Available,
            BackendAvailability::unavailable(
                BackendUnavailableReason::UnsupportedPlatform,
                "test",
            ),
            BackendAvailability::Available,
        );
        assert!(caps.is_available(Backend::Sync));
        assert!(caps.is_available(Backend::Async));
        assert!(!caps.is_available(Backend::Mmap));
        assert!(caps.is_available(Backend::IoUring));
    }

    #[test]
    fn capabilities_iter_returns_four_entries() {
        let caps = BackendCapabilities::new(
            BackendAvailability::Available,
            BackendAvailability::Available,
            BackendAvailability::Available,
            BackendAvailability::Available,
        );
        let entries = caps.iter();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].0, Backend::Sync);
        assert_eq!(entries[1].0, Backend::Async);
        assert_eq!(entries[2].0, Backend::Mmap);
        assert_eq!(entries[3].0, Backend::IoUring);
    }
}
