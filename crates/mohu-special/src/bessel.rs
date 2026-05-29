/// Bessel functions.
///
/// Implements j0, j1, jn, y0, y1, yn, i0, i1, k0, k1.
/// Uses polynomial approximations for small arguments and asymptotic expansions for large.

use std::f64::consts::PI;

/// Bessel function of the first kind, order 0: J₀(x)
#[inline(always)]
pub fn j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        // Polynomial approximation for small arguments
        const Y: [f64; 5] = [
            57568490574.0, -13362590354.0, 651619640.7, -11214424.18, 77392.33017,
        ];
        const P: [f64; 5] = [
            57568490411.0, 1029532985.0, 9494680.718, 59272.64853, 267.8532712,
        ];
        let xx = x * x;
        let num = Y[0] + xx * (Y[1] + xx * (Y[2] + xx * (Y[3] + xx * Y[4])));
        let den = P[0] + xx * (P[1] + xx * (P[2] + xx * (P[3] + xx * P[4])));
        num / den
    } else {
        // Asymptotic expansion for large arguments
        let z = 8.0 / ax;
        let y = z * z;
        const XX: [f64; 5] = [
            0.0, -0.1098626272763, 0.273451040742, -0.186556688902, 0.047625629756,
        ];
        const YY: [f64; 5] = [
            -0.024578922102, 0.37409396274, -0.839498720872, 0.905513367069, -0.337995015525,
        ];
        let num = XX[0] + y * (XX[1] + y * (XX[2] + y * (XX[3] + y * XX[4])));
        let den = YY[0] + y * (YY[1] + y * (YY[2] + y * (YY[3] + y * YY[4])));
        (0.7978845608028654 * (ax - 0.7853981633974483).cos() / ax) * (1.0 + z * num / den)
    }
}

/// Bessel function of the first kind, order 1: J₁(x)
#[inline(always)]
pub fn j1(x: f64) -> f64 {
    let ax = x.abs();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    
    if ax < 8.0 {
        // Polynomial approximation for small arguments
        const Y: [f64; 5] = [
            72362614232.0, -7895059235.0, 242396853.1, -2972611.439, 15704.48260,
        ];
        const P: [f64; 5] = [
            144725228442.0, 2300535178.0, 18586604.55, 99447.43394, 376.9991397,
        ];
        let xx = x * x;
        let num = Y[0] + xx * (Y[1] + xx * (Y[2] + xx * (Y[3] + xx * Y[4])));
        let den = P[0] + xx * (P[1] + xx * (P[2] + xx * (P[3] + xx * P[4])));
        sign * x * (num / den)
    } else {
        // Asymptotic expansion for large arguments
        let z = 8.0 / ax;
        let y = z * z;
        const XX: [f64; 5] = [
            0.0, 0.24575201774, -0.5543898132, 0.3291929800, -0.0720391779,
        ];
        const YY: [f64; 5] = [
            0.0475140389, -0.4096818257, 0.8392492628, -0.9059050145, 0.3379950155,
        ];
        let num = XX[0] + y * (XX[1] + y * (XX[2] + y * (XX[3] + y * XX[4])));
        let den = YY[0] + y * (YY[1] + y * (YY[2] + y * (YY[3] + y * YY[4])));
        sign * 0.7978845608028654 * ((ax - 2.356194491923087).cos() / ax) * (1.0 + z * num / den)
    }
}

/// Bessel function of the first kind, order n: Jₙ(x)
#[inline(always)]
pub fn jn(n: i32, x: f64) -> f64 {
    if n == 0 {
        return j0(x);
    }
    if n == 1 {
        return j1(x);
    }
    
    // Use Miller's algorithm for higher orders
    let mut j = [0.0f64; 100];
    let m = (n as usize).min(99);
    
    // Backward recurrence
    let mut j0 = 0.0;
    let mut j1 = 1.0;
    let mut scale = 1.0;
    
    for i in (0..=m).rev() {
        j[i] = scale * j0;
        let temp = j1;
        j1 = j0;
        j0 = temp + (2.0 * (i as f64 + 1.0) / x) * j1;
        
        if i == 0 {
            scale = 1.0 / j0;
        }
    }
    
    scale * j[m]
}

/// Bessel function of the second kind, order 0: Y₀(x)
#[inline(always)]
pub fn y0(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    
    let ax = x.abs();
    if ax < 8.0 {
        // Polynomial approximation for small arguments
        const Y: [f64; 5] = [
            -2957821389.0, 7062834065.0, -512359803.6, 10879881.29, -86327.92757,
        ];
        const P: [f64; 5] = [
            40076544269.0, 745249964.8, 7189466.438, 47447.26470, 226.1030244,
        ];
        let xx = x * x;
        let num = Y[0] + xx * (Y[1] + xx * (Y[2] + xx * (Y[3] + xx * Y[4])));
        let den = P[0] + xx * (P[1] + xx * (P[2] + xx * (P[3] + xx * P[4])));
        (num / den) + 0.6366197723675814 * j0(x) * x.ln()
    } else {
        // Asymptotic expansion for large arguments
        let z = 8.0 / ax;
        let y = z * z;
        const XX: [f64; 5] = [
            0.0, -0.1098626272763, 0.273451040742, -0.186556688902, 0.047625629756,
        ];
        const YY: [f64; 5] = [
            -0.024578922102, 0.37409396274, -0.839498720872, 0.905513367069, -0.337995015525,
        ];
        let num = XX[0] + y * (XX[1] + y * (XX[2] + y * (XX[3] + y * XX[4])));
        let den = YY[0] + y * (YY[1] + y * (YY[2] + y * (YY[3] + y * YY[4])));
        0.7978845608028654 * ((ax - 0.7853981633974483).sin() / ax) * (1.0 + z * num / den)
    }
}

/// Bessel function of the second kind, order 1: Y₁(x)
#[inline(always)]
pub fn y1(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    
    let ax = x.abs();
    if ax < 8.0 {
        // Polynomial approximation for small arguments
        const Y: [f64; 5] = [
            -470674184.0, 98120744.0, -10298035.79, 479439.5443, -9837.475515,
        ];
        const P: [f64; 5] = [
            249958057.0, 37252490.0, 2276778.774, 76587.58835, 1549.903665,
        ];
        let xx = x * x;
        let num = Y[0] + xx * (Y[1] + xx * (Y[2] + xx * (Y[3] + xx * Y[4])));
        let den = P[0] + xx * (P[1] + xx * (P[2] + xx * (P[3] + xx * P[4])));
        (x * (num / den)) + 0.6366197723675814 * (j1(x) * x.ln() - 1.0 / x)
    } else {
        // Asymptotic expansion for large arguments
        let z = 8.0 / ax;
        let y = z * z;
        const XX: [f64; 5] = [
            0.0, 0.24575201774, -0.5543898132, 0.3291929800, -0.0720391779,
        ];
        const YY: [f64; 5] = [
            0.0475140389, -0.4096818257, 0.8392492628, -0.9059050145, 0.3379950155,
        ];
        let num = XX[0] + y * (XX[1] + y * (XX[2] + y * (XX[3] + y * XX[4])));
        let den = YY[0] + y * (YY[1] + y * (YY[2] + y * (YY[3] + y * YY[4])));
        0.7978845608028654 * ((ax - 2.356194491923087).sin() / ax) * (1.0 + z * num / den)
    }
}

/// Bessel function of the second kind, order n: Yₙ(x)
#[inline(always)]
pub fn yn(n: i32, x: f64) -> f64 {
    if n == 0 {
        return y0(x);
    }
    if n == 1 {
        return y1(x);
    }
    
    // Use forward recurrence for Yₙ
    let mut ynm2 = y0(x);
    let mut ynm1 = y1(x);
    
    for i in 1..n {
        let yn = (2.0 * (i as f64) / x) * ynm1 - ynm2;
        ynm2 = ynm1;
        ynm1 = yn;
    }
    
    ynm1
}

/// Modified Bessel function of the first kind, order 0: I₀(x)
#[inline(always)]
pub fn i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        // Polynomial approximation for small arguments
        let y = (x / 3.75).powi(2);
        const P: [f64; 7] = [
            1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813,
        ];
        let mut result = P[0];
        for i in 1..7 {
            result += P[i] * y.powi(i as i32);
        }
        result
    } else {
        // Asymptotic expansion for large arguments
        let y = 3.75 / ax;
        const Q: [f64; 9] = [
            0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281, -0.02057706,
            0.02635537, -0.01647633, 0.00392377,
        ];
        let mut result = Q[0];
        for i in 1..9 {
            result += Q[i] * y.powi(i as i32);
        }
        result * (ax).exp() / ax.sqrt()
    }
}

/// Modified Bessel function of the first kind, order 1: I₁(x)
#[inline(always)]
pub fn i1(x: f64) -> f64 {
    let ax = x.abs();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    
    if ax < 3.75 {
        // Polynomial approximation for small arguments
        let y = (x / 3.75).powi(2);
        const P: [f64; 7] = [
            0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532, 0.00032411,
        ];
        let mut result = P[0];
        for i in 1..7 {
            result += P[i] * y.powi(i as i32);
        }
        sign * ax * result
    } else {
        // Asymptotic expansion for large arguments
        let y = 3.75 / ax;
        const Q: [f64; 9] = [
            0.39894228, -0.03988024, -0.00362018, 0.00163801, -0.01031555, 0.02282967,
            -0.02895312, 0.01787654, -0.00420059,
        ];
        let mut result = Q[0];
        for i in 1..9 {
            result += Q[i] * y.powi(i as i32);
        }
        sign * result * (ax).exp() / ax.sqrt()
    }
}

/// Modified Bessel function of the second kind, order 0: K₀(x)
#[inline(always)]
pub fn k0(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    
    if x < 2.0 {
        // Polynomial approximation for small arguments
        let y = (x / 2.0).powi(2);
        const P: [f64; 7] = [
            0.0, 0.34488577, 0.34324392, 0.11509534, 0.02153824, 0.00260555, 0.00013251,
        ];
        const Q: [f64; 7] = [
            1.0, 0.11508938, 0.06152837, 0.01508263, 0.00249015, 0.00027501, 0.00002033,
        ];
        let mut num = P[0];
        let mut den = Q[0];
        for i in 1..7 {
            num += P[i] * y.powi(i as i32);
            den += Q[i] * y.powi(i as i32);
        }
        (-x).ln() * i0(x) + num / den
    } else {
        // Asymptotic expansion for large arguments
        let y = 2.0 / x;
        const P: [f64; 7] = [
            1.0, 0.12500031, 0.00732934, 0.00024809, 0.00000595, 0.00000011, 0.00000000,
        ];
        const Q: [f64; 7] = [
            1.0, -0.37500042, -0.04687500, -0.00265208, -0.00008562, -0.00000188, -0.00000003,
        ];
        let mut num = P[0];
        let mut den = Q[0];
        for i in 1..7 {
            num += P[i] * y.powi(i as i32);
            den += Q[i] * y.powi(i as i32);
        }
        (PI / (2.0 * x)).sqrt() * (-x).exp() * num / den
    }
}

/// Modified Bessel function of the second kind, order 1: K₁(x)
#[inline(always)]
pub fn k1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    
    if x < 2.0 {
        // Polynomial approximation for small arguments
        let y = (x / 2.0).powi(2);
        const P: [f64; 7] = [
            1.0, 0.37851464, 0.17272776, 0.04151939, 0.00628119, 0.00059525, 0.00003052,
        ];
        const Q: [f64; 7] = [
            1.0, 0.20284245, 0.06388853, 0.01129687, 0.00129209, 0.00009491, 0.00000424,
        ];
        let mut num = P[0];
        let mut den = Q[0];
        for i in 1..7 {
            num += P[i] * y.powi(i as i32);
            den += Q[i] * y.powi(i as i32);
        }
        (-x).ln() * i1(x) + (1.0 / x) + num / (x * den)
    } else {
        // Asymptotic expansion for large arguments
        let y = 2.0 / x;
        const P: [f64; 7] = [
            1.0, 0.37500042, 0.04687500, 0.00265208, 0.00008562, 0.00000188, 0.00000003,
        ];
        const Q: [f64; 7] = [
            1.0, -0.12500031, -0.00732934, -0.00024809, -0.00000595, -0.00000011, -0.00000000,
        ];
        let mut num = P[0];
        let mut den = Q[0];
        for i in 1..7 {
            num += P[i] * y.powi(i as i32);
            den += Q[i] * y.powi(i as i32);
        }
        (PI / (2.0 * x)).sqrt() * (-x).exp() * num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j0() {
        assert!((j0(0.0) - 1.0).abs() < 1e-10);
        assert!((j0(1.0) - 0.7651976865579666).abs() < 1e-10);
    }

    #[test]
    fn test_j1() {
        assert!((j1(0.0) - 0.0).abs() < 1e-10);
        assert!((j1(1.0) - 0.4400505857449335).abs() < 1e-10);
    }

    #[test]
    fn test_i0() {
        assert!((i0(0.0) - 1.0).abs() < 1e-10);
        assert!((i0(1.0) - 1.266065877752008).abs() < 1e-10);
    }
}

