/// Universal-function (ufunc) protocol for mohu.
///
/// A `Ufunc` is a typed, broadcast-aware function that operates element-wise
/// over N-dimensional arrays.  This crate defines the protocol — the trait
/// definitions, loop machinery, and dispatch tables — that `mohu-ops` uses
/// to implement every arithmetic, comparison, and transcendental operation.
///
/// # Core concepts
///
/// | Concept        | Description                                              |
/// |----------------|----------------------------------------------------------|
/// | `UfuncKind`    | Unary / Binary / Generalized (multiple in/out arrays)    |
/// | `Loop`         | Inner kernel loop over a single contiguous chunk         |
/// | `TypeResolver` | Selects the output dtype from input dtypes               |
/// | `Ufunc` trait  | Ties together loop, resolver, and method set             |
/// | `UfuncMethod`  | `__call__`, `reduce`, `accumulate`, `outer`, `at`        |
///
/// # Broadcasting
///
/// All ufuncs go through the broadcast engine in [`broadcast`] before
/// dispatching to the inner loop.  The broadcast engine:
/// 1. Validates that input shapes are broadcast-compatible.
/// 2. Computes the output shape.
/// 3. Iterates over slices of the output in Rayon-parallel chunks.
///
/// # Reduce / accumulate
///
/// `reduce` collapses one axis; `accumulate` returns a running reduction.
/// Both are implemented generically in [`reduce`] and specialised per ufunc.
///
/// # Adding a new ufunc
///
/// Implement [`Ufunc`] and register it in the dispatch table.  The macro
/// [`define_ufunc!`] generates the boilerplate for common binary/unary cases.
pub mod broadcast;
pub mod dispatch;
pub mod loop_impl;
pub mod macros;
pub mod methods;
pub mod reduce;
pub mod resolver;
pub mod traits;

pub use mohu_error::{MohuError, MohuResult};
