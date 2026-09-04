/// PRNG engines and statistical distributions for mohu.
///
/// PRNG engines and statistical distributions for mohu.
///
/// The seeded convenience generator is deterministic for the selected engine
/// implementation; it does not promise stability across engine changes.
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
/// Low-level engines expose explicit seeding and stateful streams.
pub mod continuous;
pub mod discrete;
pub mod entropy;
pub mod generator;
pub mod multivariate;
pub mod permutation;
pub mod seeding;

use mohu_buffer::{buffer::Buffer, layout::Order};
use mohu_dtype::dtype::DType;

pub use generator::{Generator, Pcg64, Philox4x64};
pub use mohu_error::{MohuError, MohuResult};

/// Stateful convenience generator backed by the default `Pcg64` engine.
///
/// Two instances created with the same seed produce the same sequence for the
/// same calls with this engine implementation.
pub struct Rng {
    pcg: Pcg64,
}

impl Rng {
    /// Creates a generator with a deterministic seed.
    pub fn new(seed: u64) -> Self {
        Self {
            pcg: Pcg64::seed(seed),
        }
    }

    fn rand_f64(&mut self) -> f64 {
        let bits = self.pcg.next_u64() >> 11;
        bits as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Generates an F64 buffer with values in `[low, high)`.
    pub fn uniform(&mut self, shape: &[usize], low: f64, high: f64) -> MohuResult<Buffer> {
        if !low.is_finite() || !high.is_finite() || high <= low || !(high - low).is_finite() {
            return Err(MohuError::DomainError {
                op: "uniform",
                reason: "finite bounds required with high > low".into(),
            });
        }
        let mut output = Buffer::alloc(DType::F64, shape, Order::C)?;
        for value in output.as_mut_slice::<f64>()? {
            *value = low + (high - low) * self.rand_f64();
        }
        Ok(output)
    }

    /// Generates an F64 buffer using the Box-Muller normal transform.
    pub fn normal(&mut self, shape: &[usize], mean: f64, std: f64) -> MohuResult<Buffer> {
        if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
            return Err(MohuError::DomainError {
                op: "normal",
                reason: "finite mean and positive finite std required".into(),
            });
        }
        let mut output = Buffer::alloc(DType::F64, shape, Order::C)?;
        let values = output.as_mut_slice::<f64>()?;
        let mut i = 0;
        while i < values.len() {
            let mut u1 = self.rand_f64();
            while u1 == 0.0 {
                u1 = self.rand_f64();
            }
            let u2 = self.rand_f64();
            let magnitude = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            values[i] = mean + std * magnitude * theta.cos();
            if i + 1 < values.len() {
                values[i + 1] = mean + std * magnitude * theta.sin();
            }
            i += 2;
        }
        Ok(output)
    }

    /// Generates an I64 buffer with values in `[low, high)` without modulo bias.
    pub fn integers(&mut self, shape: &[usize], low: i64, high: i64) -> MohuResult<Buffer> {
        if high <= low {
            return Err(MohuError::DomainError {
                op: "integers",
                reason: "high must be greater than low".into(),
            });
        }
        let span = (high as i128 - low as i128) as u128;
        let domain = 1u128 << 64;
        let limit = domain - domain % span;
        let mut output = Buffer::alloc(DType::I64, shape, Order::C)?;
        for value in output.as_mut_slice::<i64>()? {
            let draw = loop {
                let draw = self.pcg.next_u64() as u128;
                if draw < limit {
                    break draw;
                }
            };
            *value = (low as i128 + (draw % span) as i128) as i64;
        }
        Ok(output)
    }
}
