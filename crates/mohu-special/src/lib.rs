pub mod bessel;
/// Special mathematical functions for mohu.
///
/// The currently implemented surface is the stable elementary helper subset
/// in [`misc`]. The remaining families are planned and are not yet equivalent
/// to SciPy or covered by the long-term numerical accuracy target.
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
/// # Implementation status
///
/// Implemented: `misc::{log1p, expm1, logit, expit, xlogy, xlog1py}` using
/// stable standard-library primitives where available.
///
/// Planned/incomplete: `erf`, `gamma`, `beta`, `bessel`, `expint`, `trig`, and
/// distribution helpers. Future implementations require documented domain,
/// pole, NaN/infinity behavior, reference vectors, boundary tests, and
/// evidence before making ULP or SciPy-compatibility claims.
pub mod beta;
pub mod erf;
pub mod expint;
pub mod gamma;
pub mod misc;
pub mod stats_fn;
pub mod trig;

pub use mohu_error::{MohuError, MohuResult};
