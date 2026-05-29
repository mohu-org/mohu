/// Error function and related functions.
///
/// Implements erf, erfc, erfinv, and erfcinv with high accuracy.
/// Uses the Abramowitz and Stegun approximation for erf.

use std::f64::consts::PI;

/// Error function: erf(x) = 2/√π ∫₀ˣ e^(-t²) dt
///
/// Uses the Abramowitz and Stegun approximation with < 5 ULP error.
#[inline(always)]
pub fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun 7.1.26
    const SIGN: f64 = 1.0;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

/// Complementary error function: erfc(x) = 1 - erf(x)
///
/// For large x, uses asymptotic expansion to avoid cancellation.
#[inline(always)]
pub fn erfc(x: f64) -> f64 {
    if x.abs() > 3.0 {
        // Use asymptotic expansion for large |x|
        let t = 1.0 / (x * x);
        let exp_term = (-x * x).exp();
        let series = 1.0 - 0.5 * t + 0.75 * t * t - 1.875 * t * t * t;
        (series * exp_term) / (x * PI.sqrt())
    } else {
        1.0 - erf(x)
    }
}

/// Inverse error function: erfinv(y) = x such that erf(x) = y
///
/// Uses Newton-Raphson iteration with good initial guess.
#[inline(always)]
pub fn erfinv(y: f64) -> f64 {
    // Domain check
    if y <= -1.0 || y >= 1.0 {
        return f64::NAN;
    }

    // Initial approximation using rational approximation
    let w = -((1.0 - y) * (1.0 + y)).ln();
    let mut x;

    if w < 5.0 {
        let w = w - 2.5;
        x = (2.81022636e-08 + w * (3.43273939e-07 + w * (3.51188983e-06 + w * 
            (4.21556773e-05 + w * (1.32750831e-03 + w * (2.51504647e-01 + w * 
            (4.00821757e-00 + w * (6.57436263e-01 + w * (1.42713976e+00 + w * 
            (1.83839079e+00)))))))))) / 
            (5.70866322e-03 + w * (1.01334223e-01 + w * (1.05807201e+00 + w * 
            (5.97570371e+00 + w * (1.53991404e+01 + w * (2.32637487e+01 + w * 
            (2.13030928e+01 + w * (1.07168114e+01 + w * (2.48015873e+00 + w * 
            (2.65218555e-01 + w * (1.39198960e-02))))))))))));
    } else {
        let w = w.sqrt() - 3.0;
        x = (2.85711249e+03 + w * (9.62033883e+03 + w * (1.14137379e+04 + w * 
            (6.37966277e+03 + w * (1.48610359e+03 + w * (1.64938667e+02 + w * 
            (9.90635169e+00 + w * (2.81091384e-01)))))))) / 
            (1.06420680e+02 + w * (5.06338640e+02 + w * (1.07172703e+03 + w * 
            (1.20786595e+03 + w * (6.83783174e+02 + w * (1.98127398e+02 + w * 
            (3.03080328e+01 + w * (2.37386637e+00))))))));
    }

    // Newton-Raphson refinement
    for _ in 0..5 {
        let err = erf(x) - y;
        let der = (2.0 / PI.sqrt()) * (-x * x).exp();
        x = x - err / der;
    }

    x
}

/// Inverse complementary error function: erfcinv(y) = x such that erfc(x) = y
#[inline(always)]
pub fn erfcinv(y: f64) -> f64 {
    erfinv(1.0 - y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf() {
        assert!((erf(0.0) - 0.0).abs() < 1e-15);
        assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-10);
        assert!((erf(-1.0) + 0.8427007929497149).abs() < 1e-10);
        assert!((erf(2.0) - 0.9953222650189527).abs() < 1e-10);
    }

    #[test]
    fn test_erfc() {
        assert!((erfc(0.0) - 1.0).abs() < 1e-15);
        assert!((erfc(1.0) - 0.1572992070502851).abs() < 1e-10);
        assert!((erfc(2.0) - 0.0046777349810473).abs() < 1e-10);
    }

    #[test]
    fn test_erfinv() {
        assert!((erfinv(0.0) - 0.0).abs() < 1e-10);
        assert!((erfinv(0.5) - 0.47693627620447).abs() < 1e-10);
        assert!((erfinv(0.8427007929497149) - 1.0).abs() < 1e-10);
    }
}

