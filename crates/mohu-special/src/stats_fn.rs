/// Statistical functions.
///
/// Implements ndtr, ndtri, chdtr, fdtr, stdtr, gdtr.
/// Uses the error function and beta function for CDF/PPF calculations.

use std::f64::consts::PI;
use super::erf::{erf, erfinv};
use super::beta::betainc;

/// Normal distribution CDF: ndtr(x) = 0.5 * (1 + erf(x/√2))
#[inline(always)]
pub fn ndtr(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / (2.0_f64).sqrt()))
}

/// Normal distribution PPF (inverse CDF): ndtri(p) = x such that ndtr(x) = p
#[inline(always)]
pub fn ndtri(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    
    erfinv(2.0 * p - 1.0) * (2.0_f64).sqrt()
}

/// Chi-squared distribution CDF: chdtr(df, x)
#[inline(always)]
pub fn chdtr(df: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    
    // Use regularized gamma function
    // P(a, x) = γ(a, x) / Γ(a)
    let a = df / 2.0;
    let x_half = x / 2.0;
    
    // For now, use a simple approximation
    // In production, this should use the regularized gamma function
    if x_half < a {
        // Series expansion for small x
        let mut result = 0.0;
        let mut term = (-x_half).exp();
        let mut i = 0;
        while i < 100 {
            result += term;
            term *= x_half / (a + i as f64 + 1.0);
            i += 1;
            if term.abs() < 1e-15 {
                break;
            }
        }
        result
    } else {
        // Continued fraction for large x
        let mut result = 1.0;
        let mut term = 1.0;
        let mut i = 0;
        while i < 100 {
            term *= x_half / (a + i as f64);
            result += term;
            i += 1;
            if term.abs() < 1e-15 {
                break;
            }
        }
        1.0 - (-x_half).exp() * result
    }
}

/// F-distribution CDF: fdtr(df1, df2, x)
#[inline(always)]
pub fn fdtr(df1: f64, df2: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    
    // Use regularized incomplete beta function
    let a = df1 / 2.0;
    let b = df2 / 2.0;
    let z = (df1 * x) / (df1 * x + df2);
    
    betainc(a, b, z)
}

/// Student's t-distribution CDF: stdtr(df, x)
#[inline(always)]
pub fn stdtr(df: f64, x: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    
    if x < 0.0 {
        return 1.0 - stdtr(df, -x);
    }
    
    let a = df / 2.0;
    let b = 0.5;
    let z = df / (df + x * x);
    
    0.5 * betainc(a, b, z)
}

/// Gamma distribution CDF: gdtr(a, b, x)
#[inline(always)]
pub fn gdtr(a: f64, b: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    
    let z = b * x;
    
    // Use regularized gamma function
    // For now, use a simple approximation
    if z < a {
        // Series expansion for small z
        let mut result = 0.0;
        let mut term = (-z).exp();
        let mut i = 0;
        while i < 100 {
            result += term;
            term *= z / (a + i as f64 + 1.0);
            i += 1;
            if term.abs() < 1e-15 {
                break;
            }
        }
        result
    } else {
        // Continued fraction for large z
        let mut result = 1.0;
        let mut term = 1.0;
        let mut i = 0;
        while i < 100 {
            term *= z / (a + i as f64);
            result += term;
            i += 1;
            if term.abs() < 1e-15 {
                break;
            }
        }
        1.0 - (-z).exp() * result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndtr() {
        assert!((ndtr(0.0) - 0.5).abs() < 1e-15);
        assert!((ndtr(1.0) - 0.8413447460685429).abs() < 1e-10);
        assert!((ndtr(-1.0) - 0.15865525393145707).abs() < 1e-10);
    }

    #[test]
    fn test_ndtri() {
        assert!((ndtri(0.5) - 0.0).abs() < 1e-10);
        assert!((ndtri(0.8413447460685429) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fdtr() {
        assert!((fdtr(1.0, 1.0, 1.0) - 0.5).abs() < 1e-10);
    }
}

