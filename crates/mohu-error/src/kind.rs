use crate::codes::ErrorCode;

/// A coarse four-way classification of every `MohuError`.
///
/// Use `ErrorKind` when you need to branch on broad error category —
/// for example, to decide whether to retry, log at `warn` vs `error`,
/// or surface a different UI message — without matching every individual
/// `MohuError` variant or `ErrorCode`.
///
/// # Example
///
/// ```rust
/// # use mohu_error::{MohuError, kind::ErrorKind};
/// fn handle(e: &MohuError) {
///     match e.kind() {
///         ErrorKind::Usage    => eprintln!("caller error: {e}"),
///         ErrorKind::Runtime  => eprintln!("runtime failure: {e}"),
///         ErrorKind::System   => eprintln!("system/IO failure: {e}"),
///         ErrorKind::Internal => panic!("mohu bug: {e}"),
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ErrorKind {
    /// The caller passed invalid arguments — wrong shapes, out-of-bounds
    /// indices, incompatible dtypes, etc.  These errors indicate a
    /// programming mistake and should never be retried as-is.
    Usage = 0,

    /// A well-formed operation failed at runtime due to the mathematical
    /// properties of the data — singular matrix, non-convergence,
    /// domain error, etc.
    Runtime = 1,

    /// A system-level failure outside mohu's control — I/O, memory
    /// allocation, DLPack version mismatch, Arrow IPC failure.
    System = 2,

    /// An invariant inside mohu was violated.  These should never appear
    /// in production and always indicate a bug in mohu itself.
    Internal = 3,
}

impl ErrorKind {
    /// Returns `true` if this kind represents a recoverable situation
    /// (i.e. retrying might succeed without changing the inputs).
    ///
    /// Currently only `System` errors are considered potentially
    /// recoverable, since I/O and allocation failures can be transient.
    pub fn is_recoverable(self) -> bool {
        matches!(self, Self::System)
    }

    /// Human-readable one-word label for this kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::Usage => "usage",
            Self::Runtime => "runtime",
            Self::System => "system",
            Self::Internal => "internal",
        }
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Maps an `ErrorCode` to its broad `ErrorKind`.
impl From<ErrorCode> for ErrorKind {
    fn from(code: ErrorCode) -> Self {
        match code as u32 {
            // Shape, DType, Index — all caller mistakes
            1000..=3999 => ErrorKind::Usage,

            // Buffer — mostly caller mistakes (bad strides, read-only),
            // but allocation failure is a system error.
            4000..=4002 => ErrorKind::System, // Alloc, Align, BufSmall
            4003..=4999 => ErrorKind::Usage,

            // Compute — runtime mathematical failures
            5000..=5999 => ErrorKind::Runtime,

            // I/O — system
            6000..=6999 => ErrorKind::System,

            // DLPack — mostly usage (wrong device, bad version)
            // except null pointer which is an internal invariant violation
            7002 => ErrorKind::Internal, // DLPackNullPointer
            7000..=7999 => ErrorKind::Usage,

            // Arrow — system/IPC
            8000..=8999 => ErrorKind::System,

            // Python — usage
            9000..=9999 => ErrorKind::Usage,

            // Context: delegate to inner error — handled in MohuError::kind()
            // NotImplemented: runtime
            // Internal: internal
            10000 => ErrorKind::Runtime, // Context (placeholder; overridden)
            10001 => ErrorKind::Runtime, // NotImplemented
            10002 => ErrorKind::Internal, // Internal

            _ => ErrorKind::Internal,
        }
    }
}
