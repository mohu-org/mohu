//! Stable elementary scalar helpers.

/// Computes `ln(1 + x)` without losing precision for small `x`.
///
/// # Example
///
/// ```
/// assert_eq!(mohu_special::misc::log1p(0.0), 0.0);
/// ```
#[inline]
pub fn log1p(x: f64) -> f64 {
    x.ln_1p()
}

/// Computes `exp(x) - 1` without losing precision for small `x`.
///
/// # Example
///
/// ```
/// assert_eq!(mohu_special::misc::expm1(0.0), 0.0);
/// ```
#[inline]
pub fn expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// Computes the logit function on the real unit interval.
///
/// Values outside `[0, 1]` produce `NaN`; endpoints produce signed infinities.
///
/// # Example
///
/// ```
/// assert_eq!(mohu_special::misc::logit(0.5), 0.0);
/// ```
#[inline]
pub fn logit(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    p.ln() - (-p).ln_1p()
}

/// Computes the numerically stable logistic function.
///
/// # Example
///
/// ```
/// assert_eq!(mohu_special::misc::expit(0.0), 0.5);
/// ```
#[inline]
pub fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Computes `x * ln(y)`, defining `xlogy(0, y)` as zero.
///
/// # Example
///
/// ```
/// assert_eq!(mohu_special::misc::xlogy(0.0, 0.0), 0.0);
/// ```
#[inline]
pub fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

/// Computes `x * ln(1 + y)`, defining `xlog1py(0, y)` as zero.
///
/// # Example
///
/// ```
/// assert_eq!(mohu_special::misc::xlog1py(0.0, -1.0), 0.0);
/// ```
#[inline]
pub fn xlog1py(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln_1p() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_log_helpers_cover_boundaries() {
        assert_eq!(log1p(0.0), 0.0);
        assert_eq!(log1p(-0.0), -0.0);
        assert!(log1p(-1.0).is_infinite());
        assert!(log1p(-1.1).is_nan());
        assert_eq!(expm1(0.0), 0.0);
        assert_eq!(expm1(-0.0), -0.0);
        assert_eq!(expm1(f64::NEG_INFINITY), -1.0);
        assert!(expm1(f64::NAN).is_nan());
    }
    #[test]
    fn logistic_and_logit_are_stable() {
        assert_eq!(expit(0.0), 0.5);
        assert_eq!(expit(f64::INFINITY), 1.0);
        assert_eq!(expit(f64::NEG_INFINITY), 0.0);
        assert!(expit(f64::NAN).is_nan());
        assert_eq!(logit(0.0), f64::NEG_INFINITY);
        assert_eq!(logit(1.0), f64::INFINITY);
        assert!(logit(-0.1).is_nan());
        assert!(logit(1.1).is_nan());
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            assert!((logit(expit(x)) - x).abs() < 1e-12);
        }
    }
    #[test]
    fn xlog_helpers_preserve_zero_and_domains() {
        assert_eq!(xlogy(0.0, 0.0), 0.0);
        assert_eq!(xlog1py(0.0, -1.0), 0.0);
        assert!(xlogy(1.0, -1.0).is_nan());
        assert!(xlog1py(1.0, -1.1).is_nan());
        assert!(xlogy(1.0, 0.0).is_infinite());
        assert!(xlog1py(1.0, -1.0).is_infinite());
    }
}
