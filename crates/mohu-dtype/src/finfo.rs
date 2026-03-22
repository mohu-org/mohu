/// Floating-point type metadata — the Rust equivalent of `numpy.finfo`.
///
/// Every field mirrors NumPy's `finfo` object exactly so Python code that
/// inspects `np.finfo(np.float32)` can find the same values via
/// `FloatInfo::of(DType::F32)`.
///
/// # Example
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, finfo::FloatInfo};
/// let info = FloatInfo::of(DType::F32).unwrap();
/// println!("f32 epsilon = {}", info.eps);        // 1.1920929e-7
/// println!("f32 bits    = {}", info.bits);        // 32
/// println!("f32 nmant   = {}", info.nmant);       // 23
/// println!("f32 maxexp  = {}", info.maxexp);      // 128
/// ```
use mohu_error::{MohuError, MohuResult};

use crate::dtype::DType;

/// Machine-precision metadata for a floating-point `DType`.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatInfo {
    /// The `DType` this info describes.
    pub dtype: DType,

    /// Total number of bits in the representation.
    pub bits: u32,

    /// Number of mantissa (significand) bits, *excluding* the hidden leading 1.
    /// F16 = 10, BF16 = 7, F32 = 23, F64 = 52.
    pub nmant: u32,

    /// Number of exponent bits.
    /// F16 = 5, BF16 = 8, F32 = 8, F64 = 11.
    pub nexp: u32,

    /// Maximum (biased) exponent value + 1.  The largest integer power of 2
    /// representable.  F16 = 16, BF16 = 128, F32 = 128, F64 = 1024.
    pub maxexp: i32,

    /// Minimum (biased) exponent value.  The smallest integer power of 2
    /// for a normalised number.  F16 = -13, BF16 = -125, F32 = -125, F64 = -1021.
    pub minexp: i32,

    /// Machine epsilon: the smallest positive value such that `1.0 + eps != 1.0`.
    /// This is `2^{-nmant}` (one ULP at 1.0).
    pub eps: f64,

    /// Largest finite representable value.
    pub max: f64,

    /// Smallest finite representable value (= `-max`).
    pub min: f64,

    /// Smallest positive *normalised* value (NumPy calls this `tiny`).
    pub tiny: f64,

    /// Same as `tiny` — provided for NumPy API compat (`finfo.smallest_normal`).
    pub smallest_normal: f64,

    /// Smallest positive *subnormal* value.
    pub smallest_subnormal: f64,

    /// Number of significant decimal digits: `floor(nmant * log10(2))`.
    pub precision: u32,

    /// Decimal resolution: `10^{-precision}`.
    pub resolution: f64,

    /// Epsilon used for allclose comparisons: `max(eps, tiny)`.
    pub epsneg: f64,
}

impl FloatInfo {
    /// Returns `FloatInfo` for the given dtype, or an error if the dtype is
    /// not a floating-point type.
    pub fn of(dtype: DType) -> MohuResult<Self> {
        match dtype {
            DType::F16  => Ok(Self::f16()),
            DType::BF16 => Ok(Self::bf16()),
            DType::F32  => Ok(Self::f32()),
            DType::F64  => Ok(Self::f64()),
            other => Err(MohuError::UnsupportedDType {
                op: "finfo",
                dtype: other.to_string(),
            }),
        }
    }

    // ─── F16 ──────────────────────────────────────────────────────────────────

    /// Returns `FloatInfo` for IEEE 754 binary16 (half-precision).
    pub fn f16() -> Self {
        // IEEE 754 binary16: sign=1, exp=5, mantissa=10
        let eps             = half::f16::EPSILON.to_f64();
        let max             = half::f16::MAX.to_f64();
        let tiny            = half::f16::MIN_POSITIVE.to_f64();
        // Smallest subnormal: 2^{-24}
        let smallest_sub    = 5.960_464_477_539_063e-8_f64;
        let precision       = (10_u32 as f64 * f64::log10(2.0)).floor() as u32; // 3
        Self {
            dtype:             DType::F16,
            bits:              16,
            nmant:             10,
            nexp:              5,
            maxexp:            16,
            minexp:            -13,
            eps,
            max,
            min:               -max,
            tiny,
            smallest_normal:   tiny,
            smallest_subnormal: smallest_sub,
            precision,
            resolution:        10f64.powi(-(precision as i32)),
            epsneg:            eps,
        }
    }

    // ─── BF16 ─────────────────────────────────────────────────────────────────

    /// Returns `FloatInfo` for Google Brain Float16 (bfloat16).
    pub fn bf16() -> Self {
        // Google Brain Float16: sign=1, exp=8, mantissa=7
        let eps             = half::bf16::EPSILON.to_f64();
        let max             = half::bf16::MAX.to_f64();
        let tiny            = half::bf16::MIN_POSITIVE.to_f64();
        let smallest_sub    = 9.183_549_615_799_121e-41_f64;
        let precision       = (7_u32 as f64 * f64::log10(2.0)).floor() as u32; // 2
        Self {
            dtype:             DType::BF16,
            bits:              16,
            nmant:             7,
            nexp:              8,
            maxexp:            128,
            minexp:            -125,
            eps,
            max,
            min:               -max,
            tiny,
            smallest_normal:   tiny,
            smallest_subnormal: smallest_sub,
            precision,
            resolution:        10f64.powi(-(precision as i32)),
            epsneg:            eps,
        }
    }

    // ─── F32 ──────────────────────────────────────────────────────────────────

    /// Returns `FloatInfo` for IEEE 754 binary32 (single-precision).
    pub fn f32() -> Self {
        let eps             = f32::EPSILON as f64;
        let max             = f32::MAX as f64;
        let tiny            = f32::MIN_POSITIVE as f64;
        let smallest_sub    = 1.401_298_464_324_817e-45_f64;
        let precision       = (23_u32 as f64 * f64::log10(2.0)).floor() as u32; // 6
        Self {
            dtype:             DType::F32,
            bits:              32,
            nmant:             23,
            nexp:              8,
            maxexp:            128,
            minexp:            -125,
            eps,
            max,
            min:               -max,
            tiny,
            smallest_normal:   tiny,
            smallest_subnormal: smallest_sub,
            precision,
            resolution:        10f64.powi(-(precision as i32)),
            epsneg:            eps,
        }
    }

    // ─── F64 ──────────────────────────────────────────────────────────────────

    /// Returns `FloatInfo` for IEEE 754 binary64 (double-precision).
    pub fn f64() -> Self {
        let eps             = f64::EPSILON;
        let max             = f64::MAX;
        let tiny            = f64::MIN_POSITIVE;
        let smallest_sub    = 5.0e-324_f64;
        let precision       = (52_u32 as f64 * f64::log10(2.0)).floor() as u32; // 15
        Self {
            dtype:             DType::F64,
            bits:              64,
            nmant:             52,
            nexp:              11,
            maxexp:            1024,
            minexp:            -1021,
            eps,
            max,
            min:               -max,
            tiny,
            smallest_normal:   tiny,
            smallest_subnormal: smallest_sub,
            precision,
            resolution:        10f64.powi(-(precision as i32)),
            epsneg:            eps,
        }
    }

    // ─── convenience ──────────────────────────────────────────────────────────

    /// Returns `true` if `|a - b| <= atol + rtol * |b|` (NumPy `isclose`
    /// default tolerances based on this dtype's epsilon).
    pub fn isclose_default_tol(&self) -> (f64, f64) {
        // rtol = eps^{2/3}, atol = tiny
        let rtol = self.eps.powf(2.0 / 3.0);
        (rtol, self.tiny)
    }

    /// Returns the ULP (unit in the last place) at value `x`.
    pub fn ulp_at(&self, x: f64) -> f64 {
        // ULP(x) ≈ |x| * eps * 2  (for normalised x)
        x.abs() * self.eps * 2.0
    }

    /// Returns `true` if `|a - b| <= n_ulps * ULP(max(|a|, |b|))`.
    pub fn within_ulps(&self, a: f64, b: f64, n_ulps: u64) -> bool {
        let scale = a.abs().max(b.abs());
        if scale == 0.0 { return a == b; }
        let ulp = self.ulp_at(scale);
        (a - b).abs() <= n_ulps as f64 * ulp
    }
}

impl std::fmt::Display for FloatInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FloatInfo({dtype}: bits={bits}, nmant={nmant}, nexp={nexp}, eps={eps:.3e}, max={max:.3e}, tiny={tiny:.3e})",
            dtype = self.dtype,
            bits  = self.bits,
            nmant = self.nmant,
            nexp  = self.nexp,
            eps   = self.eps,
            max   = self.max,
            tiny  = self.tiny,
        )
    }
}
