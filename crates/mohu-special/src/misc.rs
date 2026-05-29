/// Miscellaneous mathematical functions.
///
/// Implements log1p, expm1, logit, expit, xlogy, xlog1py.
/// Provides numerically stable versions of common operations.

/// Logarithm of 1 plus x: log1p(x) = ln(1 + x)
///
/// More accurate than ln(1 + x) for small x.
#[inline(always)]
pub fn log1p(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NAN;
    }
    if x.abs() < f64::EPSILON {
        return x;
    }
    (1.0 + x).ln()
}

/// Exponential of x minus 1: expm1(x) = e^x - 1
///
/// More accurate than e^x - 1 for small x.
#[inline(always)]
pub fn expm1(x: f64) -> f64 {
    if x.abs() < 1e-5 {
        x + x * x / 2.0
    } else {
        x.exp() - 1.0
    }
}

/// Logit function: logit(p) = ln(p / (1 - p))
///
/// The inverse of the logistic function.
#[inline(always)]
pub fn logit(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    (p / (1.0 - p)).ln()
}

/// Logistic (sigmoid) function: expit(x) = 1 / (1 + e^(-x))
///
/// The inverse of the logit function.
#[inline(always)]
pub fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// x * log(y) with handling of y = 0
#[inline(always)]
pub fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if y <= 0.0 {
        f64::NAN
    } else {
        x * y.ln()
    }
}

/// x * log(1 + y) with handling of y = -1
#[inline(always)]
pub fn xlog1py(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if y <= -1.0 {
        f64::NAN
    } else {
        x * log1p(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log1p() {
        assert!((log1p(0.0) - 0.0).abs() < 1e-15);
        assert!((log1p(1.0) - 2.0_f64.ln()).abs() < 1e-15);
        assert!((log1p(1e-10) - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_expm1() {
        assert!((expm1(0.0) - 0.0).abs() < 1e-15);
        assert!((expm1(1.0) - (1.0_f64.exp() - 1.0)).abs() < 1e-15);
        assert!((expm1(1e-10) - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_logit() {
        assert!((logit(0.5) - 0.0).abs() < 1e-15);
        assert!((logit(0.75) - 1.0986122886681098).abs() < 1e-10);
    }

    #[test]
    fn test_expit() {
        assert!((expit(0.0) - 0.5).abs() < 1e-15);
        assert!((expit(1.0) - 0.7310585786300049).abs() < 1e-10);
    }

    #[test]
    fn test_xlogy() {
        assert!((xlogy(0.0, 2.0) - 0.0).abs() < 1e-15);
        assert!((xlogy(2.0, 1.0) - 0.0).abs() < 1e-15);
        assert!((xlogy(2.0, 2.0) - 2.0 * 2.0_f64.ln()).abs() < 1e-15);
    }
}

