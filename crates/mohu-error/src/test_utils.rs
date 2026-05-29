/// Testing utilities for code that returns `MohuResult<T>`.
///
/// These are available in `#[cfg(test)]` blocks everywhere in the
/// workspace via `use mohu_error::test_utils::*`.
///
/// # Assertion macros
///
/// - [`assert_err!`] — assert that a result is `Err(_)`
/// - [`assert_ok!`] — assert that a result is `Ok(_)` and return the value
/// - [`assert_err_code!`] — assert the exact error code
/// - [`assert_err_kind!`] — assert the broad error kind
/// - [`assert_shape_err!`] — assert a specific shape mismatch
/// - [`assert_err_chain!`] — assert the depth of the context chain
use crate::{MohuError, MohuResult, codes::ErrorCode, kind::ErrorKind};

// ─── assertion helpers (non-macro) ───────────────────────────────────────────

/// Asserts that `result` is `Err(_)` and returns the inner `MohuError`.
///
/// Panics with a descriptive message if `result` is `Ok(_)`.
///
/// # Example
///
/// ```rust
/// # use mohu_error::{MohuError, MohuResult, test_utils::assert_err};
/// let r: MohuResult<()> = Err(MohuError::DivisionByZero);
/// let err = assert_err(r, "should fail on zero divisor");
/// assert!(matches!(err, MohuError::DivisionByZero));
/// ```
pub fn assert_err<T: std::fmt::Debug>(result: MohuResult<T>, context: &str) -> MohuError {
    match result {
        Err(e) => e,
        Ok(v) => panic!("assert_err failed ({context}): expected Err(_), got Ok({v:?})"),
    }
}

/// Asserts that `result` is `Ok(_)` and returns the inner value.
///
/// Panics with a descriptive message including the error if `result` is `Err`.
pub fn assert_ok<T>(result: MohuResult<T>, context: &str) -> T {
    match result {
        Ok(v) => v,
        Err(e) => panic!("assert_ok failed ({context}): expected Ok(_), got Err({e})"),
    }
}

/// Asserts that `result` is `Err` with exactly the given `ErrorCode`.
///
/// Returns the `MohuError` on success for further assertions.
pub fn assert_err_code<T: std::fmt::Debug>(
    result: MohuResult<T>,
    expected_code: ErrorCode,
    context: &str,
) -> MohuError {
    let err = assert_err(result, context);
    let actual = err.root_cause().code();
    if actual != expected_code {
        panic!(
            "assert_err_code failed ({context}): \
             expected code {expected_code}, got {actual}\n\
             error was: {err}"
        );
    }
    err
}

/// Asserts that `result` is `Err` with the given `ErrorKind`.
///
/// Returns the `MohuError` on success for further assertions.
pub fn assert_err_kind<T: std::fmt::Debug>(
    result: MohuResult<T>,
    expected_kind: ErrorKind,
    context: &str,
) -> MohuError {
    let err = assert_err(result, context);
    let actual = err.root_cause().kind();
    if actual != expected_kind {
        panic!(
            "assert_err_kind failed ({context}): \
             expected kind {expected_kind:?}, got {actual:?}\n\
             error was: {err}"
        );
    }
    err
}

/// Asserts that the error is a `ShapeMismatch` with the given shapes.
pub fn assert_shape_err<T: std::fmt::Debug>(
    result: MohuResult<T>,
    expected_shape: &[usize],
    got_shape: &[usize],
) {
    let err = assert_err(result, "expected ShapeMismatch");
    let root = err.root_cause();
    match root {
        MohuError::ShapeMismatch { expected, got } => {
            if expected.as_slice() != expected_shape || got.as_slice() != got_shape {
                panic!(
                    "assert_shape_err: shapes don't match.\n\
                     expected ShapeMismatch {{ expected: {expected_shape:?}, got: {got_shape:?} }}\n\
                     got      ShapeMismatch {{ expected: {expected:?}, got: {got:?} }}"
                );
            }
        },
        other => panic!("assert_shape_err: expected ShapeMismatch, got {other:?}"),
    }
}

/// Asserts that the error chain is exactly `depth` levels deep.
///
/// Depth 0 = no `Context` wrappers.
pub fn assert_chain_depth<T: std::fmt::Debug>(
    result: MohuResult<T>,
    depth: usize,
    context: &str,
) -> MohuError {
    let err = assert_err(result, context);
    let actual = err.chain_depth();
    if actual != depth {
        panic!(
            "assert_chain_depth failed ({context}): \
             expected chain depth {depth}, got {actual}\n\
             error was: {err}"
        );
    }
    err
}

// ─── macro versions ───────────────────────────────────────────────────────────

/// Assert that a `MohuResult` is `Err`, returning the error.
///
/// ```rust
/// # use mohu_error::{MohuError, MohuResult, assert_err};
/// let r: MohuResult<i32> = Err(MohuError::DivisionByZero);
/// let e = assert_err!(r);
/// assert!(matches!(e, MohuError::DivisionByZero));
/// ```
#[macro_export]
macro_rules! assert_err {
    ($result:expr) => {
        $crate::test_utils::assert_err($result, stringify!($result))
    };
    ($result:expr, $ctx:expr) => {
        $crate::test_utils::assert_err($result, $ctx)
    };
}

/// Assert that a `MohuResult` is `Ok`, returning the value.
///
/// ```rust
/// # use mohu_error::{MohuResult, assert_ok};
/// let r: MohuResult<i32> = Ok(42);
/// let v = assert_ok!(r);
/// assert_eq!(v, 42);
/// ```
#[macro_export]
macro_rules! assert_ok {
    ($result:expr) => {
        $crate::test_utils::assert_ok($result, stringify!($result))
    };
    ($result:expr, $ctx:expr) => {
        $crate::test_utils::assert_ok($result, $ctx)
    };
}

/// Assert that a `MohuResult` is `Err` with the exact `ErrorCode`.
///
/// ```rust
/// # use mohu_error::{MohuError, MohuResult, codes::ErrorCode, assert_err_code};
/// let r: MohuResult<()> = Err(MohuError::DivisionByZero);
/// assert_err_code!(r, ErrorCode::DivisionByZero);
/// ```
#[macro_export]
macro_rules! assert_err_code {
    ($result:expr, $code:expr) => {
        $crate::test_utils::assert_err_code($result, $code, stringify!($result))
    };
}

/// Assert that a `MohuResult` is `Err` with the given `ErrorKind`.
#[macro_export]
macro_rules! assert_err_kind {
    ($result:expr, $kind:expr) => {
        $crate::test_utils::assert_err_kind($result, $kind, stringify!($result))
    };
}
