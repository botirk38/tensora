use std::fmt;

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

}
