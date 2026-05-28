pub mod chain;
pub mod codes;
pub mod context;
pub mod error;
pub mod kind;
pub mod macros;
pub mod multi;
pub mod reporter;
pub mod test_utils;

#[cfg(feature = "python")]
pub mod python;

pub use chain::ErrorChain;
pub use codes::ErrorCode;
pub use context::ResultExt;
pub use error::MohuError;
pub use kind::ErrorKind;
pub use multi::MultiError;
pub use reporter::{ErrorReporter, ReportMode, Severity};

/// The universal result type used by every mohu crate.
pub type MohuResult<T> = std::result::Result<T, MohuError>;
