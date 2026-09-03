/// SIMD kernel primitives for mohu.
///
/// This crate provides hand-written AVX2, AVX-512, and NEON kernels for the
/// hot paths in `mohu-ops` and `mohu-ufunc`.  The public API is a set of
/// free functions that accept raw pointers and lengths; callers are
/// responsible for alignment and length invariants.
///
/// # Feature flags
///
/// | Feature            | ISA enabled              |
/// |--------------------|--------------------------|
/// | `avx2`             | x86-64 AVX2 (256-bit)    |
/// | `avx512`           | x86-64 AVX-512 (512-bit) |
/// | `neon`             | AArch64 NEON             |
/// | `runtime-dispatch` | pick best ISA at runtime |
///
/// When no feature flag is set, every kernel falls back to a scalar
/// implementation that the auto-vectoriser can still optimise.
///
/// # Kernel families
///
/// | Module        | Operations                                      |
/// |---------------|-------------------------------------------------|
/// | [`arith`]     | add, sub, mul, div, neg, abs, min, max          |
/// | [`cmp`]       | eq, ne, lt, le, gt, ge                          |
/// | [`reduce`]    | sum, product, min, max, mean (parallel tree)    |
/// | [`cast`]      | SIMD-accelerated element-wise type casts         |
/// | [`fill`]      | broadcast-fill a buffer with a scalar value      |
/// | [`copy`]      | SIMD memcpy with non-temporal stores for large  |
/// | [`math`]      | sqrt, rsqrt, exp, log, sin, cos (approx + exact)|
/// | [`fma`]       | fused multiply-add / multiply-subtract           |
/// | [`bitwise`]   | and, or, xor, not, shl, shr for integer types   |
pub mod arith;
pub mod bitwise;
pub mod cast;
pub mod cmp;
pub mod copy;
pub mod fill;
pub mod fma;
pub mod math;
pub mod reduce;

/// Runtime CPU feature detection.
pub mod detect;

pub use mohu_error::{MohuError, MohuResult};
