pub mod bessel;
/// Special mathematical functions for mohu.
///
/// Equivalent to `scipy.special` — pure-Rust implementations with
/// double-precision accuracy (< 1 ULP for most functions) and SIMD-ready
/// scalar kernels that `mohu-ops` can vectorise.
///
/// # Function families
///
/// | Module        | Functions                                              |
/// |---------------|--------------------------------------------------------|
/// | [`erf`]       | erf, erfc, erfinv, erfcinv                             |
/// | [`gamma`]     | gamma, lgamma, digamma, polygamma, rgamma              |
/// | [`beta`]      | beta, lbeta, betainc, betaincinv                       |
/// | [`bessel`]    | j0, j1, jn, y0, y1, yn, i0, i1, k0, k1                |
/// | [`expint`]    | expn, e1, ei                                           |
/// | [`trig`]      | sinc, sindg, cosdg, cotdg                              |
/// | [`stats_fn`]  | ndtr, ndtri, chdtr, fdtr, stdtr, gdtr (CDF/PPF)        |
/// | [`misc`]      | log1p, expm1, logit, expit, xlogy, xlog1py             |
///
/// # Accuracy targets
///
/// All functions aim for < 5 ULP error on the standard IEEE double range.
/// Where the underlying algorithm cannot achieve this, the docstring
/// documents the actual error bound.
///
/// # Vectorisation
///
/// Every scalar function is `#[inline(always)]` and designed to auto-vectorise
/// under LLVM.  `mohu-simd` provides hand-written AVX2 versions for the
/// most common (erf, gamma, exp, log) on x86-64.
pub mod beta;
pub mod erf;
pub mod expint;
pub mod gamma;
pub mod misc;
pub mod stats_fn;
pub mod trig;

pub use mohu_error::{MohuError, MohuResult};
