/// Trigonometric functions.
///
/// Implements sinc, sindg, cosdg, cotdg.
/// Uses standard trigonometric functions with degree conversions.

use std::f64::consts::PI;

/// Sinc function: sinc(x) = sin(πx) / (πx)
#[inline(always)]
pub fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let pi_x = PI * x;
    pi_x.sin() / pi_x
}

/// Sine of degrees: sindg(x) = sin(x * π/180)
#[inline(always)]
pub fn sindg(x: f64) -> f64 {
    (x * PI / 180.0).sin()
}

/// Cosine of degrees: cosdg(x) = cos(x * π/180)
#[inline(always)]
pub fn cosdg(x: f64) -> f64 {
    (x * PI / 180.0).cos()
}

/// Cotangent of degrees: cotdg(x) = cot(x * π/180)
#[inline(always)]
pub fn cotdg(x: f64) -> f64 {
    let rad = x * PI / 180.0;
    let sin_val = rad.sin();
    if sin_val == 0.0 {
        return if rad.cos() > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    rad.cos() / sin_val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinc() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-15);
        assert!((sinc(1.0) - 0.0).abs() < 1e-15);
        assert!((sinc(0.5) - 0.6366197723675814).abs() < 1e-10);
    }

    #[test]
    fn test_sindg() {
        assert!((sindg(0.0) - 0.0).abs() < 1e-15);
        assert!((sindg(90.0) - 1.0).abs() < 1e-10);
        assert!((sindg(30.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cosdg() {
        assert!((cosdg(0.0) - 1.0).abs() < 1e-15);
        assert!((cosdg(90.0) - 0.0).abs() < 1e-10);
        assert!((cosdg(60.0) - 0.5).abs() < 1e-10);
    }
}

