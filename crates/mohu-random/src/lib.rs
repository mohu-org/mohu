/// PRNG engines and statistical distributions for mohu.
///
/// Equivalent to `numpy.random` — but faster, thread-safe, and reproducible
/// across platforms.  Every generator uses a splittable / jumpable PRNG so
/// that parallel workers each get an independent, non-overlapping stream.
///
/// # Generators
///
/// | Type         | Algorithm     | Notes                                 |
/// |--------------|---------------|---------------------------------------|
/// | `Pcg64`      | PCG-64-DXSM   | Default — fast, statistically strong  |
/// | `Philox4x64` | Philox 4×64   | Counter-based, GPU-friendly           |
/// | `ChaCha8`    | ChaCha8       | Cryptographically secure              |
/// | `SplitMix64` | SplitMix64    | Lightweight, used for seeding         |
///
/// # Distributions
///
/// | Module          | Distributions                                       |
/// |-----------------|-----------------------------------------------------|
/// | [`continuous`]  | uniform, normal, standard_t, gamma, beta, chi2, …   |
/// | [`discrete`]    | integers, binomial, poisson, geometric, hypergeom   |
/// | [`multivariate`]| multivariate_normal, dirichlet, multinomial          |
/// | [`permutation`] | shuffle, permutation, choice                        |
///
/// # Reproducibility
///
/// ```rust,ignore
/// let mut rng = mohu_random::Pcg64::seed(42);
/// let data = rng.standard_normal::<f64>(&[1000]);
/// ```
///
/// All generators implement `Seed` — the same seed always produces the
/// same sequence regardless of CPU count or mohu version within a major.
pub mod continuous;
pub mod discrete;
pub mod entropy;
pub mod generator;
pub mod multivariate;
pub mod permutation;
pub mod seeding;

pub use generator::{Generator, Pcg64, Philox4x64};
pub use mohu_error::{MohuError, MohuResult};
