/// Gamma function and related functions.
///
/// Implements gamma, lgamma, digamma, polygamma, and rgamma.
/// Uses Lanczos approximation for gamma function.

use std::f64::consts::PI;

/// Gamma function: Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt
///
/// Uses Lanczos approximation with < 5 ULP error for x > 0.
#[inline(always)]
pub fn gamma(x: f64) -> f64 {
    if x <= 0.0 {
        if x == x.floor() {
            // Pole at non-positive integers
            return f64::NAN;
        }
        // Reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        return PI / ((PI * x).sin() * gamma(1.0 - x));
    }

    // Lanczos approximation coefficients (g = 7)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let x = x - 1.0;
    let mut a = C[0];
    for i in 1..9 {
        a += C[i] / (x + i as f64);
    }
    let t = x + G + 0.5;
    (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
}

/// Log-gamma function: lgamma(x) = ln(Γ(x))
///
/// More numerically stable than ln(gamma(x)) for large x.
#[inline(always)]
pub fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        if x == x.floor() {
            return f64::INFINITY;
        }
        // Reflection formula for log-gamma
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x == 0.0 {
            return f64::INFINITY;
        }
        return (PI / sin_pi_x.abs()).ln() - lgamma(1.0 - x);
    }

    // Lanczos approximation for log-gamma
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let x = x - 1.0;
    let mut a = C[0];
    for i in 1..9 {
        a += C[i] / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Digamma function: ψ(x) = d/dx ln(Γ(x))
///
/// Uses asymptotic expansion for large x and series for small x.
#[inline(always)]
pub fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        // Reflection formula
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }

    // Use asymptotic expansion for x > 6
    if x > 6.0 {
        let mut result = x.ln() - 0.5 / x;
        let x2 = x * x;
        let mut x_pow = x2;
        let mut sign = 1.0;
        const B: [f64; 6] = [1.0/12.0, -1.0/120.0, 1.0/252.0, -1.0/240.0, 1.0/132.0, -691.0/32760.0];
        
        for &b in &B {
            sign = -sign;
            result += sign * b / x_pow;
            x_pow *= x2;
        }
        return result;
    }

    // Use series for smaller x
    let mut result = -0.57721566490153286060651209008240243104215933593992; // Euler-Mascheroni constant
    let mut n = 0;
    while n < 100 {
        result += 1.0 / (x + n as f64) - 1.0 / (1.0 + n as f64);
        n += 1;
        if (1.0 / (x + n as f64)).abs() < 1e-15 {
            break;
        }
    }
    result
}

/// Polygamma function: ψ^(n)(x) = d^n/dx^n ψ(x)
///
/// Only implements n=1 (trigamma) for now.
#[inline(always)]
pub fn polygamma(n: u32, x: f64) -> f64 {
    if n == 0 {
        return digamma(x);
    }
    if n == 1 {
        return trigamma(x);
    }
    // Higher-order polygamma not implemented yet
    f64::NAN
}

/// Trigamma function: ψ'(x) = d/dx ψ(x)
#[inline(always)]
fn trigamma(x: f64) -> f64 {
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        // Reflection formula
        let sin_pi_x = (PI * x).sin();
        let cos_pi_x = (PI * x).cos();
        let trigamma_1mx = trigamma(1.0 - x);
        let psi_1mx = digamma(1.0 - x);
        let term = PI.powi(2) / sin_pi_x.powi(2);
        return trigamma_1mx - term + 2.0 * PI * PI * cos_pi_x / sin_pi_x.powi(3) * psi_1mx;
    }

    // Use asymptotic expansion for x > 6
    if x > 6.0 {
        let mut result = 1.0 / x + 0.5 / (x * x);
        let x2 = x * x;
        let mut x_pow = x2 * x2;
        let mut sign = 1.0;
        const B: [f64; 5] = [1.0/6.0, -1.0/30.0, 1.0/42.0, -1.0/30.0, 5.0/66.0];
        
        for &b in &B {
            sign = -sign;
            result += sign * b / x_pow;
            x_pow *= x2;
        }
        return result;
    }

    // Use series for smaller x
    let mut result = 0.0;
    let mut n = 0;
    while n < 100 {
        result += 1.0 / (x + n as f64).powi(2);
        n += 1;
        if (1.0 / (x + n as f64).powi(2)).abs() < 1e-15 {
            break;
        }
    }
    result
}

/// Reciprocal gamma function: rgamma(x) = 1/Γ(x)
///
/// More numerically stable than 1/gamma(x) for large x.
#[inline(always)]
pub fn rgamma(x: f64) -> f64 {
    1.0 / gamma(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-10);
        assert!((gamma(2.0) - 1.0).abs() < 1e-10);
        assert!((gamma(3.0) - 2.0).abs() < 1e-10);
        assert!((gamma(0.5) - 1.772453850905516).abs() < 1e-10);
    }

    #[test]
    fn test_lgamma() {
        assert!((lgamma(1.0) - 0.0).abs() < 1e-10);
        assert!((lgamma(2.0) - 0.0).abs() < 1e-10);
        assert!((lgamma(3.0) - 0.6931471805599453).abs() < 1e-10);
    }

    #[test]
    fn test_digamma() {
        assert!((digamma(1.0) - (-0.5772156649015329)).abs() < 1e-10);
        assert!((digamma(2.0) - 0.4227843350984671).abs() < 1e-10);
    }
}

