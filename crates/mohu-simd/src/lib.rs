//! Low-level SIMD kernels with portable fallbacks.
//!
//! Fill and copy expose raw-pointer APIs with explicit safety contracts.
//!
//! AVX2 kernels are compiled only with the `avx2` Cargo feature and entered
//! only when runtime CPU detection reports AVX2 support. AVX-512 and NEON
//! kernels remain planned.
//!
//! # Kernel families
//!
//! The `fill` and `copy` modules are implemented. The other modules are
//! reserved for future arithmetic, comparison, reduction, cast, math, FMA,
//! and bitwise kernels.

pub mod arith;
pub mod bitwise;
pub mod cast;
pub mod cmp;
pub mod copy;
pub mod detect;
pub mod fill;
pub mod fma;
pub mod math;
pub mod reduce;

pub use detect::{CpuFeatures, cpu_features};
pub use mohu_error::{MohuError, MohuResult};
