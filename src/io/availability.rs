//! I/O backend availability and capability types.
//!
//! These types describe which I/O backends are available in the current
//! process environment and why a given engine may be unavailable.
//!
//! # Usage
//!
//! ```rust,ignore
//! use tensora::io::availability::Capabilities;
//!
//! let caps = Capabilities::probe();
//! if caps.is_available(BackendKind::IoUring) {
//!     // use io_uring path
//! }
//! ```

use std::fmt;
use std::sync::OnceLock;

// ============================================================================
// BackendKind
// ============================================================================

/// Identifies a I/O backend.
///
/// Sync and Tokio have explicit implementations for Linux, macOS, and Windows.
/// IoUring is Linux-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Synchronous blocking I/O.
    ///
    /// Linux: O_DIRECT-aware chunked reads with `write_at` positioned writes.
    /// macOS: `std::os::unix::fs::FileExt` positioned I/O.
    /// Windows: `std::os::windows::fs::FileExt` positioned I/O.
    Sync,
    /// Tokio async I/O.
    ///
    /// All platforms delegate reads to `Sync` via `spawn_blocking`.
    /// Writes clone the caller-supplied file and write via `spawn_blocking`.
    Tokio,
    /// Memory-mapped file access.
    Mmap,
    /// Linux io_uring multi-worker I/O (Linux only).
    ///
    /// Always reports `Unavailable` on non-Linux platforms.
    IoUring,
}

impl BackendKind {
    /// Returns a short, stable identifier string for this engine.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sync => "sync",
            Self::Tokio => "tokio",
            Self::Mmap => "mmap",
            Self::IoUring => "io-uring",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// UnavailableReason
// ============================================================================

/// Why a I/O backend cannot be used in the current environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnavailableReason {
    /// The engine requires a platform this process is not running on.
    UnsupportedPlatform,
    /// The process lacks required privileges (e.g. `CAP_SYS_ADMIN` for io_uring).
    PermissionDenied,
    /// A required kernel feature is missing (e.g. `CONFIG_IO_URING`).
    MissingKernelFeature,
    /// The kernel has the feature but it is misconfigured.
    InvalidKernelConfiguration,
    /// A required runtime dependency is missing or returned an unexpected value.
    MissingDependency,
    /// The filesystem does not support the required operation.
    FilesystemUnsupported,
    /// Any other reason; carries a human-readable explanation.
    Other(String),
}

impl UnavailableReason {
    /// Returns a short, stable code string suitable for logging or serialization.
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

impl fmt::Display for UnavailableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => f.write_str("unsupported platform"),
            Self::PermissionDenied => f.write_str("permission denied"),
            Self::MissingKernelFeature => f.write_str("missing kernel feature"),
            Self::InvalidKernelConfiguration => f.write_str("invalid kernel configuration"),
            Self::MissingDependency => f.write_str("missing dependency"),
            Self::FilesystemUnsupported => f.write_str("filesystem unsupported"),
            Self::Other(msg) => f.write_str(msg),
        }
    }
}

// ============================================================================
// Availability
// ============================================================================

/// Availability status for a single I/O backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// The engine is available and can be used.
    Available,
    /// The engine cannot be used; `reason` and `details` explain why.
    Unavailable {
        /// Structured reason code.
        reason: UnavailableReason,
        /// Human-readable details, may be empty.
        details: String,
    },
}

impl Availability {
    /// Returns `true` if the engine is available.
    #[inline]
    #[must_use]
    pub const fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }

    /// Constructs an `Unavailable` variant.
    #[must_use]
    pub fn unavailable(reason: UnavailableReason, details: impl Into<String>) -> Self {
        Self::Unavailable {
            reason,
            details: details.into(),
        }
    }
}

impl fmt::Display for Availability {
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

// ============================================================================
// Capabilities
// ============================================================================

/// Snapshot of all I/O backend availability for deterministic selection.
///
/// Obtain via [`Capabilities::probe`] at startup; cache the result
/// rather than calling `probe` on every I/O decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Sync engine availability.
    pub sync: Availability,
    /// Tokio async engine availability.
    pub tokio: Availability,
    /// Mmap engine availability.
    pub mmap: Availability,
    /// io_uring engine availability (Linux only; always `Unavailable` elsewhere).
    pub io_uring: Availability,
}

impl Capabilities {
    /// Returns the availability of the given engine.
    #[must_use]
    pub fn availability(&self, kind: BackendKind) -> &Availability {
        match kind {
            BackendKind::Sync => &self.sync,
            BackendKind::Tokio => &self.tokio,
            BackendKind::Mmap => &self.mmap,
            BackendKind::IoUring => &self.io_uring,
        }
    }

    /// Returns `true` if the given engine is available.
    #[inline]
    #[must_use]
    pub fn is_available(&self, kind: BackendKind) -> bool {
        self.availability(kind).is_available()
    }

    /// Returns all four `(kind, availability)` pairs.
    #[must_use]
    pub fn iter(&self) -> [(BackendKind, &Availability); 4] {
        [
            (BackendKind::Sync, &self.sync),
            (BackendKind::Tokio, &self.tokio),
            (BackendKind::Mmap, &self.mmap),
            (BackendKind::IoUring, &self.io_uring),
        ]
    }

    /// Probe the current environment and return a capability snapshot.
    ///
    /// Sync and Tokio are always available. Mmap availability depends on the
    /// `region` crate returning a non-zero page size. io_uring is only probed
    /// on Linux.
    #[must_use]
    pub fn probe() -> Self {
        let mmap = Self::probe_mmap();
        let io_uring = Self::probe_io_uring();
        Self {
            sync: Availability::Available,
            tokio: Availability::Available,
            mmap,
            io_uring,
        }
    }

    /// Returns a process-wide cached capability snapshot.
    ///
    /// Use this for format heuristics and hot paths. Use [`Capabilities::probe`]
    /// only when a fresh environment probe is explicitly required.
    #[must_use]
    pub fn cached() -> &'static Self {
        static CAPABILITIES: OnceLock<Capabilities> = OnceLock::new();
        CAPABILITIES.get_or_init(Self::probe)
    }

    fn probe_mmap() -> Availability {
        let page_size = region::page::size();
        if page_size == 0 {
            return Availability::unavailable(
                UnavailableReason::MissingDependency,
                "region returned a zero page size",
            );
        }
        Availability::Available
    }

    #[cfg(target_os = "linux")]
    fn probe_io_uring() -> Availability {
        // Attempt to create a minimal ring to verify the kernel supports it
        // and the process has the required privileges.
        match io_uring::IoUring::new(2) {
            Ok(_) => Availability::Available,
            Err(e) => {
                use std::io::ErrorKind;
                let reason = match e.kind() {
                    ErrorKind::PermissionDenied => UnavailableReason::PermissionDenied,
                    _ => UnavailableReason::MissingKernelFeature,
                };
                Availability::unavailable(reason, e.to_string())
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn probe_io_uring() -> Availability {
        Availability::unavailable(
            UnavailableReason::UnsupportedPlatform,
            "io_uring is only available on Linux",
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_kind_as_str() {
        assert_eq!(BackendKind::Sync.as_str(), "sync");
        assert_eq!(BackendKind::Tokio.as_str(), "tokio");
        assert_eq!(BackendKind::Mmap.as_str(), "mmap");
        assert_eq!(BackendKind::IoUring.as_str(), "io-uring");
    }

    #[test]
    fn storage_kind_display_matches_as_str() {
        for kind in [BackendKind::Sync, BackendKind::Tokio, BackendKind::Mmap, BackendKind::IoUring] {
            assert_eq!(format!("{kind}"), kind.as_str());
        }
    }

    #[test]
    fn unavailable_reason_codes() {
        assert_eq!(
            UnavailableReason::UnsupportedPlatform.code(),
            "unsupported-platform"
        );
        assert_eq!(
            UnavailableReason::PermissionDenied.code(),
            "permission-denied"
        );
        assert_eq!(
            UnavailableReason::MissingKernelFeature.code(),
            "missing-kernel-feature"
        );
        assert_eq!(
            UnavailableReason::InvalidKernelConfiguration.code(),
            "invalid-kernel-configuration"
        );
        assert_eq!(
            UnavailableReason::MissingDependency.code(),
            "missing-dependency"
        );
        assert_eq!(
            UnavailableReason::FilesystemUnsupported.code(),
            "filesystem-unsupported"
        );
        assert_eq!(UnavailableReason::Other("x".into()).code(), "other");
    }

    #[test]
    fn unavailable_reason_display() {
        let cases = [
            (
                UnavailableReason::UnsupportedPlatform,
                "unsupported platform",
            ),
            (UnavailableReason::PermissionDenied, "permission denied"),
            (
                UnavailableReason::MissingKernelFeature,
                "missing kernel feature",
            ),
            (
                UnavailableReason::InvalidKernelConfiguration,
                "invalid kernel configuration",
            ),
            (UnavailableReason::MissingDependency, "missing dependency"),
            (
                UnavailableReason::FilesystemUnsupported,
                "filesystem unsupported",
            ),
            (UnavailableReason::Other("custom msg".into()), "custom msg"),
        ];
        for (reason, expected) in cases {
            assert_eq!(format!("{reason}"), expected);
        }
    }

    #[test]
    fn storage_availability_is_available() {
        assert!(Availability::Available.is_available());
        assert!(
            !Availability::unavailable(UnavailableReason::PermissionDenied, "").is_available()
        );
    }

    #[test]
    fn storage_availability_display_available() {
        assert_eq!(Availability::Available.to_string(), "available");
    }

    #[test]
    fn storage_availability_display_unavailable_with_details() {
        let s = Availability::unavailable(
            UnavailableReason::PermissionDenied,
            "io_uring_setup returned EPERM",
        );
        assert_eq!(
            s.to_string(),
            "unavailable: permission denied (io_uring_setup returned EPERM)"
        );
    }

    #[test]
    fn storage_availability_display_unavailable_no_details() {
        let s = Availability::unavailable(UnavailableReason::UnsupportedPlatform, "");
        assert_eq!(s.to_string(), "unavailable: unsupported platform");
    }

    #[test]
    fn capabilities_probe_sync_and_tokio_always_available() {
        let caps = Capabilities::probe();
        assert!(caps.is_available(BackendKind::Sync));
        assert!(caps.is_available(BackendKind::Tokio));
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn capabilities_probe_io_uring_unavailable_on_non_linux() {
        let caps = Capabilities::probe();
        assert!(!caps.is_available(BackendKind::IoUring));
    }

    #[test]
    fn capabilities_iter_returns_four_entries() {
        let caps = Capabilities::probe();
        let entries = caps.iter();
        assert_eq!(entries.len(), 4);
        let kinds: Vec<_> = entries.iter().map(|(k, _)| *k).collect();
        assert!(kinds.contains(&BackendKind::Sync));
        assert!(kinds.contains(&BackendKind::Tokio));
        assert!(kinds.contains(&BackendKind::Mmap));
        assert!(kinds.contains(&BackendKind::IoUring));
    }
}
