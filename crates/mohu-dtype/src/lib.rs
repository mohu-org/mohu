//! Data type system for the mohu scientific computing library.
//!
//! This crate defines [`DType`], the runtime type tag for every element type
//! mohu supports, along with the [`Scalar`] trait hierarchy that provides
//! compile-time type-level operations. Together they form the bridge between
//! Rust's static type system and the dynamic typing needed for NumPy-compatible
//! array operations.
//!
//! # Key types
//!
//! | Type / Trait | Purpose |
//! |--------------|---------|
//! | [`DType`] | Runtime enum of all 15 supported element types (analogous to `numpy.dtype`) |
//! | [`Scalar`] | Sealed base trait implemented by all 15 element types |
//! | [`FloatInfo`] | Machine-precision metadata (analogous to `numpy.finfo`) |
//! | [`IntInfo`] | Integer range metadata (analogous to `numpy.iinfo`) |
//! | [`CastMode`] | Controls which casts are permitted (`Safe`, `SameKind`, `Unsafe`) |
//!
//! # Dispatch macros
//!
//! The [`dispatch_dtype!`] family of macros converts a runtime `DType` into a
//! monomorphised generic call, achieving zero-overhead dispatch without vtables.
//!
//! # Example
//!
//! ```rust
//! use mohu_dtype::{DType, promote};
//!
//! let dt = promote(DType::I32, DType::F32);
//! assert_eq!(dt, DType::F64);
//! ```

pub mod cast;
pub mod compat;
pub mod dlpack;
pub mod dtype;
pub mod finfo;
pub mod iinfo;
pub mod macros;
pub mod promote;
pub mod scalar;

pub use dtype::{DType, ALL_DTYPES, DTYPE_COUNT};
pub use finfo::FloatInfo;
pub use iinfo::IntInfo;
pub use promote::{
    can_cast, common_type, minimum_scalar_type, promote, result_type, weak_promote, CastMode,
};
pub use scalar::{
    ComplexScalar, FloatScalar, IntScalar, RealScalar, Scalar, SignedScalar, UnsignedScalar,
};

pub use mohu_error::{MohuError, MohuResult};
