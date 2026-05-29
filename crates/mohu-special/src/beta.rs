/// Beta function and related functions.
///
/// Implements beta, lbeta, betainc, and betaincinv.
/// Uses the relationship with gamma functions.

use super::gamma::{gamma, lgamma};

/// Beta function: B(a,b) = Γ(a)Γ(b) / Γ(a+b)
#[inline(always)]
pub fn beta(a: f64, b: f64) -> f64 {
    gamma(a) * gamma(b) / gamma(a + b)
}

/// Log-beta function: lbeta(a,b) = ln(B(a,b))
///
/// More numerically stable than ln(beta(a,b)) for large values.
#[inline(always)]
pub fn lbeta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Regularized incomplete beta function: I_x(a,b)
///
/// Uses continued fraction expansion for numerical stability.
#[inline(always)]
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    
    // Use continued fraction expansion
    let lbeta_ab = lbeta(a, b);
    let prefactor = (a * x.ln() + b * (1.0 - x).ln() - lbeta_ab).exp();
    
    if x < (a + 1.0) / (a + b + 2.0) {
        prefactor * betainc_continued_fraction(a, b, x) / a
    } else {
        1.0 - prefactor * betainc_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Continued fraction expansion for incomplete beta
fn betainc_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 100;
    const EPS: f64 = 1e-10;
    
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < f64::EPSILON {
        d = f64::EPSILON;
    }
    d = 1.0 / d;
    let mut h = d;
    
    for m in 1..=MAX_ITER {
        let m2 = 2 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < f64::EPSILON {
            d = f64::EPSILON;
        }
        c = 1.0 + aa / c;
        if c.abs() < f64::EPSILON {
            c = f64::EPSILON;
        }
        d = 1.0 / d;
        h *= d * c;
        
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < f64::EPSILON {
            d = f64::EPSILON;
        }
        c = 1.0 + aa / c;
        if c.abs() < f64::EPSILON {
            c = f64::EPSILON;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    
    h
}

/// Inverse regularized incomplete beta function: I_x(a,b) = p, solve for x
///
/// Uses Newton-Raphson iteration with good initial guess.
#[inline(always)]
pub fn betaincinv(a: f64, b: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    
    // Initial guess using approximation
    let mut x = if a > 1.0 && b > 1.0 {
        let t = (2.0 / (9.0 * a)).sqrt() * (1.0 - 2.0 / (9.0 * b)).sqrt();
        let y = (1.0 - t).max(0.0);
        let z = if p < 0.5 {
            -y
        } else {
            y
        };
        0.5 * (1.0 + z * (1.0 - z * z / 3.0).sqrt())
    } else if a == 1.0 {
        1.0 - (1.0 - p).powf(1.0 / b)
    } else if b == 1.0 {
        p.powf(1.0 / a)
    } else {
        0.5
    };
    
    // Newton-Raphson iteration
    for _ in 0..20 {
        let y = betainc(a, b, x) - p;
        if y.abs() < 1e-10 {
            break;
        }
        
        // Derivative using beta function
        let beta_ab = beta(a, b);
        let d = x.powf(a - 1.0) * (1.0 - x).powf(b - 1.0) / beta_ab;
        
        if d.abs() < f64::EPSILON {
            break;
        }
        
        x = (x - y / d).max(0.0).min(1.0);
    }
    
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta() {
        assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-10);
        assert!((beta(2.0, 2.0) - 1.0/6.0).abs() < 1e-10);
    }

    #[test]
    fn test_betainc() {
        assert!((betainc(1.0, 1.0, 0.5) - 0.5).abs() < 1e-10);
        assert!((betainc(2.0, 2.0, 0.5) - 0.5).abs() < 1e-10);
    }
}

