/// Testing utilities for mohu — fixtures, assertions, and property tests.
///
/// This crate is intended as a `[dev-dependency]` for every other mohu crate
/// and for downstream users who want to test code that works with mohu arrays.
///
/// # Modules
///
/// | Module        | Purpose                                               |
/// |---------------|-------------------------------------------------------|
/// | [`assert`]    | `assert_array_eq!`, `assert_allclose!`, numeric checks|
/// | [`gen`]       | `proptest` strategies: random arrays of any dtype     |
/// | [`fixtures`]  | pre-built arrays used across mohu's own test suites   |
/// | [`approx`]    | element-wise approximate equality with ULP tolerance  |
/// | [`dtype`]     | dtype-parameterised test helpers                      |
/// | [`perf`]      | micro-benchmark helpers: throughput, latency          |
///
/// # Approximate equality
///
/// ```rust,ignore
/// use mohu_testing::assert_allclose;
/// assert_allclose!(result, expected, rtol = 1e-5, atol = 1e-8);
/// ```
///
/// # Property testing
///
/// ```rust,ignore
/// use mohu_testing::gen::array_f32;
/// proptest! {
///     #[test]
///     fn add_commutative(a in array_f32([4, 4]), b in array_f32([4, 4])) {
///         assert_allclose!(a.add(&b), b.add(&a), atol = 0.0);
///     }
/// }
/// ```
pub mod approx;
pub mod assert;
pub mod dtype;
pub mod fixtures;
pub mod perf;
pub mod strategies;

pub use mohu_error::{MohuError, MohuResult};

/// Re-export `approx` crate for downstream users.
pub use approx as approx_crate;

/// Re-export `proptest` for downstream property tests.
pub use proptest;
