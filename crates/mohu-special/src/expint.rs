/// Exponential integral functions.
///
/// Implements expn, e1, and ei.
/// Uses continued fraction expansion and series expansions.

/// Exponential integral Eₙ(x) = ∫₁^∞ e^(-xt) / tⁿ dt
#[inline(always)]
pub fn expn(n: i32, x: f64) -> f64 {
    if n < 0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if n == 1 { f64::INFINITY } else { 1.0 / (n as f64 - 1.0) };
    }
    if n == 0 {
        return (-x).exp() / x;
    }
    
    // Use continued fraction expansion
    let n_f = n as f64;
    let mut b = x + n_f;
    let mut c = 1.0 / f64::MIN_POSITIVE;
    let mut d = b;
    let mut h = d;
    
    for i in 1..100 {
        let a = -(i as f64) * ((n as f64) + i as f64 - 1.0);
        b += 2.0;
        d = a + b * d;
        if d.abs() < f64::MIN_POSITIVE {
            d = f64::MIN_POSITIVE;
        }
        c = b + a / c;
        if c.abs() < f64::MIN_POSITIVE {
            c = f64::MIN_POSITIVE;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-10 {
            break;
        }
    }
    
    h * (-x).exp() / x
}

/// Exponential integral E₁(x) = ∫ₓ^∞ e^(-t) / t dt
#[inline(always)]
pub fn e1(x: f64) -> f64 {
    if x <= 0.0 {
        if x == 0.0 {
            return f64::INFINITY;
        }
        return f64::NAN;
    }
    
    if x < 1.0 {
        // Series expansion for small x
        let psi = 0.57721566490153286060651209008240243104215933593992; // Euler-Mascheroni constant
        let mut result = -psi - x.ln();
        let mut term = -x;
        let mut i = 1;
        while i < 100 {
            result += term / i as f64;
            term *= -x / (i as f64 + 1.0);
            i += 1;
            if term.abs() < 1e-15 {
                break;
            }
        }
        result
    } else {
        // Asymptotic expansion for large x
        expn(1, x)
    }
}

/// Exponential integral Ei(x) = -∫₋ₓ^∞ e^(-t) / t dt
#[inline(always)]
pub fn ei(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    
    if x < 0.0 {
        // For negative x, use relationship with E₁
        -e1(-x)
    } else if x < 1.0 {
        // Series expansion for small positive x
        let psi = 0.57721566490153286060651209008240243104215933593992;
        let mut result = psi + x.ln();
        let mut term = x;
        let mut i = 1;
        while i < 100 {
            result += term / i as f64;
            term *= x / (i as f64 + 1.0);
            i += 1;
            if term.abs() < 1e-15 {
                break;
            }
        }
        result
    } else {
        // Asymptotic expansion for large x
        let mut result = (-x).exp() / x;
        let mut term = result;
        let mut i = 1;
        while i < 20 {
            term *= -i as f64 / x;
            result += term;
            i += 1;
            if term.abs() < 1e-15 {
                break;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e1() {
        assert!((e1(1.0) - 0.21938393439552027).abs() < 1e-10);
        assert!((e1(2.0) - 0.04890051070806112).abs() < 1e-10);
    }

    #[test]
    fn test_ei() {
        assert!((ei(1.0) - 1.8951178163559368).abs() < 1e-10);
        assert!((ei(2.0) - 4.954234356001890).abs() < 1e-10);
    }
}

