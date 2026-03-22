//! Unified error handling for the mohu scientific computing library.
//!
//! This crate provides [`MohuError`], a single non-exhaustive error enum that
//! every crate in the mohu workspace returns via [`MohuResult<T>`]. Errors are
//! organized by domain (shape, dtype, index, buffer, compute, I/O, DLPack,
//! Arrow, Python) with stable numeric [`ErrorCode`]s and a coarse four-way
//! [`ErrorKind`] classification.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`MohuError`] | The central error enum — every mohu function returns this |
//! | [`ErrorCode`] | Stable numeric code for programmatic branching |
//! | [`ErrorKind`] | Coarse category: `Usage`, `Runtime`, `System`, `Internal` |
//! | [`ErrorChain`] | Iterator over nested `Context` wrappers |
//! | [`MultiError`] | Accumulates multiple errors in a single pass |
//! | [`ErrorReporter`] | Rich terminal formatting (compact / full / JSON) |
//! | [`ResultExt`] | `.context()` and `.with_context()` extension trait |
//!
//! # Example
//!
//! ```rust
//! use mohu_error::{MohuResult, MohuError, bail, ensure};
//!
//! fn safe_divide(a: f64, b: f64) -> MohuResult<f64> {
//!     ensure!(b != 0.0, MohuError::DivisionByZero);
//!     Ok(a / b)
//! }
//! ```

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
