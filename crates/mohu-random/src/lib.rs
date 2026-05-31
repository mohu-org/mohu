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
/// let mut rng = mohu_random::Rng::new(42);
/// let data = rng.uniform(&[1000], 0.0, 1.0).unwrap();
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

use crate::generator::Generator;
use mohu_buffer::buffer::Buffer;
use mohu_buffer::layout::Order;
use mohu_dtype::dtype::DType;

pub use generator::{Generator, Pcg64, Philox4x64};
pub use mohu_error::{MohuError, MohuResult};

/// Convenience random number generator for common sampling tasks.
///
/// Wraps the default PCG-64-DXSM engine and provides methods for
/// generating uniform, normal, and integer random values into
/// multi-dimensional buffers.
///
/// # Reproducibility
///
/// Two `Rng` instances created with the same seed produce identical
/// output sequences regardless of platform or mohu version.
///
/// # Example
///
/// ```rust,ignore
/// let mut rng = Rng::new(42);
/// let u = rng.uniform(&[3, 4], 0.0, 1.0).unwrap();
/// let n = rng.normal(&[1000], 0.0, 1.0).unwrap();
/// let i = rng.integers(&[5], 0, 10).unwrap();
/// ```
pub struct Rng {
    pcg: Pcg64,
}

impl Rng {
    /// Create a new RNG seeded with `seed`.
    ///
    /// The same seed always produces the same sequence of values.
    pub fn new(seed: u64) -> Self {
        Self {
            pcg: Pcg64::seed(seed),
        }
    }

    /// Generate a random `f64` uniformly distributed in `[0, 1)`.
    fn rand_f64(&mut self) -> f64 {
        let bits = self.pcg.next_u64() >> 11;
        (bits as f64) * (1.0 / 9007199254740992.0)
    }

    /// Fill a buffer with uniformly distributed `f64` values.
    ///
    /// Each value lies in `[low, high)`.
    ///
    /// # Errors
    ///
    /// Returns `DomainError` if `high <= low`.
    pub fn uniform(&mut self, shape: &[usize], low: f64, high: f64) -> MohuResult<Buffer> {
        if high <= low {
            return Err(MohuError::DomainError {
                op: "uniform",
                reason: "high must be > low".into(),
            });
        }
        let mut out = Buffer::alloc(DType::F64, shape, Order::C)?;
        let slice = out.as_mut_slice::<f64>()?;
        let scale = high - low;
        for v in slice.iter_mut() {
            *v = low + scale * self.rand_f64();
        }
        Ok(out)
    }

    /// Fill a buffer with normally distributed `f64` values.
    ///
    /// Uses the Box-Muller transform to generate pairs of independent
    /// standard normal variates, then scales by `std` and shifts by `mean`.
    ///
    /// # Errors
    ///
    /// Returns `DomainError` if `std <= 0.0`.
    pub fn normal(&mut self, shape: &[usize], mean: f64, std: f64) -> MohuResult<Buffer> {
        if std <= 0.0 {
            return Err(MohuError::DomainError {
                op: "normal",
                reason: "std must be > 0".into(),
            });
        }
        let len: usize = shape.iter().product();
        let mut out = Buffer::alloc(DType::F64, shape, Order::C)?;
        let slice = out.as_mut_slice::<f64>()?;
        let mut i = 0;
        while i < len {
            let u1 = self.rand_f64();
            let u2 = self.rand_f64();
            let mag = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            slice[i] = mean + std * mag * theta.cos();
            if i + 1 < len {
                slice[i + 1] = mean + std * mag * theta.sin();
            }
            i += 2;
        }
        Ok(out)
    }

    /// Fill a buffer with uniformly distributed random integers.
    ///
    /// Each value lies in `[low, high)`.
    ///
    /// # Errors
    ///
    /// Returns `DomainError` if `high <= low`.
    pub fn integers(&mut self, shape: &[usize], low: i64, high: i64) -> MohuResult<Buffer> {
        if high <= low {
            return Err(MohuError::DomainError {
                op: "integers",
                reason: "high must be > low".into(),
            });
        }
        let range = (high - low) as u64;
        let mut out = Buffer::alloc(DType::I64, shape, Order::C)?;
        let slice = out.as_mut_slice::<i64>()?;
        for v in slice.iter_mut() {
            *v = low + (self.pcg.next_u64() % range) as i64;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_range() {
        let mut rng = Rng::new(42);
        let buf = rng.uniform(&[1000], 0.0, 1.0).unwrap();
        let slice = buf.as_slice::<f64>().unwrap();
        for &v in slice.iter() {
            assert!((0.0..1.0).contains(&v), "value {} out of [0,1)", v);
        }
    }

    #[test]
    fn test_uniform_wider_range() {
        let mut rng = Rng::new(99);
        let buf = rng.uniform(&[500], -5.0, 5.0).unwrap();
        let slice = buf.as_slice::<f64>().unwrap();
        for &v in slice.iter() {
            assert!((-5.0..5.0).contains(&v), "value {} out of [-5,5)", v);
        }
    }

    #[test]
    fn test_uniform_invalid_range() {
        let mut rng = Rng::new(1);
        assert!(rng.uniform(&[1], 5.0, 3.0).is_err());
        assert!(rng.uniform(&[1], 2.0, 2.0).is_err());
    }

    #[test]
    fn test_normal_shape() {
        let mut rng = Rng::new(7);
        let buf = rng.normal(&[3, 4, 5], 0.0, 1.0).unwrap();
        assert_eq!(buf.shape(), &[3, 4, 5]);
        assert_eq!(buf.dtype(), DType::F64);
    }

    #[test]
    fn test_normal_mean_std() {
        let mut rng = Rng::new(123);
        let n = 100_000;
        let buf = rng.normal(&[n], 2.0, 3.0).unwrap();
        let slice = buf.as_slice::<f64>().unwrap();
        let mean = slice.iter().sum::<f64>() / n as f64;
        let variance = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_est = variance.sqrt();
        // Allow 2% relative tolerance for n=100000
        assert!(
            (mean - 2.0).abs() < 0.1,
            "mean {} too far from 2.0",
            mean
        );
        assert!(
            (std_est - 3.0).abs() < 0.1,
            "std {} too far from 3.0",
            std_est
        );
    }

    #[test]
    fn test_normal_invalid_std() {
        let mut rng = Rng::new(1);
        assert!(rng.normal(&[1], 0.0, 0.0).is_err());
        assert!(rng.normal(&[1], 0.0, -1.0).is_err());
    }

    #[test]
    fn test_integers_range() {
        let mut rng = Rng::new(42);
        let buf = rng.integers(&[1000], 0, 10).unwrap();
        let slice = buf.as_slice::<i64>().unwrap();
        for &v in slice.iter() {
            assert!((0..10).contains(&v), "value {} out of [0,10)", v);
        }
    }

    #[test]
    fn test_integers_negative_low() {
        let mut rng = Rng::new(7);
        let buf = rng.integers(&[500], -5, 5).unwrap();
        let slice = buf.as_slice::<i64>().unwrap();
        for &v in slice.iter() {
            assert!((-5..5).contains(&v), "value {} out of [-5,5)", v);
        }
    }

    #[test]
    fn test_integers_invalid_range() {
        let mut rng = Rng::new(1);
        assert!(rng.integers(&[1], 5, 3).is_err());
        assert!(rng.integers(&[1], 2, 2).is_err());
    }

    #[test]
    fn test_reproducibility() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        let a = rng1.uniform(&[100], 0.0, 1.0).unwrap();
        let b = rng2.uniform(&[100], 0.0, 1.0).unwrap();
        assert_eq!(a.as_slice::<f64>().unwrap(), b.as_slice::<f64>().unwrap());
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut rng1 = Rng::new(1);
        let mut rng2 = Rng::new(2);
        let a = rng1.uniform(&[10], 0.0, 1.0).unwrap();
        let b = rng2.uniform(&[10], 0.0, 1.0).unwrap();
        // Extremely unlikely to be equal
        assert_ne!(a.as_slice::<f64>().unwrap(), b.as_slice::<f64>().unwrap());
    }

    #[test]
    fn test_empty_shape() {
        let mut rng = Rng::new(0);
        let buf = rng.uniform(&[0], 0.0, 1.0).unwrap();
        assert_eq!(buf.len(), 0);
        let buf = rng.normal(&[0, 5], 0.0, 1.0).unwrap();
        assert_eq!(buf.len(), 0);
        let buf = rng.integers(&[0], 0, 10).unwrap();
        assert_eq!(buf.len(), 0);
    }
}
