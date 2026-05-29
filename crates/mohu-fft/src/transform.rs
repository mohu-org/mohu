use num_complex::Complex;
use rustfft::{FftPlanner, num_complex::Complex as RComplex};

use crate::Norm;

/// Compute the 1-D FFT of `input` with optional length `n` and `norm` mode.
/// If `n` is larger than `input.len()` the input is zero-padded; if smaller,
/// it is truncated.
pub fn fft(input: &[Complex<f64>], n: Option<usize>, norm: Norm) -> Vec<Complex<f64>> {
    let len = n.unwrap_or(input.len());
    if len == 0 {
        return Vec::new();
    }
    let mut buf: Vec<RComplex<f64>> = vec![RComplex::new(0.0, 0.0); len];
    for (i, v) in input.iter().take(len).enumerate() {
        buf[i] = RComplex::new(v.re, v.im);
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(len);
    fft.process(&mut buf);

    // apply forward normalization
    let scale = match norm {
        Norm::Backward => 1.0,
        Norm::Ortho => 1.0 / (len as f64).sqrt(),
        Norm::Forward => 1.0 / (len as f64),
    };

    buf.into_iter()
        .map(|c| Complex::new(c.re * scale, c.im * scale))
        .collect()
}

/// Compute the 1-D inverse FFT (IFFT) of `input` with optional length `n` and `norm` mode.
pub fn ifft(input: &[Complex<f64>], n: Option<usize>, norm: Norm) -> Vec<Complex<f64>> {
    let len = n.unwrap_or(input.len());
    if len == 0 {
        return Vec::new();
    }
    let mut buf: Vec<RComplex<f64>> = vec![RComplex::new(0.0, 0.0); len];
    for (i, v) in input.iter().take(len).enumerate() {
        buf[i] = RComplex::new(v.re, v.im);
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(len);
    ifft.process(&mut buf);

    // apply backward normalization
    let scale = match norm {
        Norm::Backward => 1.0 / (len as f64),
        Norm::Ortho => 1.0 / (len as f64).sqrt(),
        Norm::Forward => 1.0,
    };

    buf.into_iter()
        .map(|c| Complex::new(c.re * scale, c.im * scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mohu_testing::assert_allclose;
    use num_complex::Complex;

    fn assert_complex_close(actual: &[Complex<f64>], expected: &[Complex<f64>]) {
        let actual_re: Vec<f64> = actual.iter().map(|value| value.re).collect();
        let actual_im: Vec<f64> = actual.iter().map(|value| value.im).collect();
        let expected_re: Vec<f64> = expected.iter().map(|value| value.re).collect();
        let expected_im: Vec<f64> = expected.iter().map(|value| value.im).collect();

        assert_allclose!(actual_re, expected_re, atol = 1e-9);
        assert_allclose!(actual_im, expected_im, atol = 1e-9);
    }

    #[test]
    fn roundtrip_fft_ifft_backward() {
        let input: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let out = fft(&input, None, Norm::Backward);
        let back = ifft(&out, None, Norm::Backward);
        assert_complex_close(&back, &input);
    }

    #[test]
    fn roundtrip_fft_ifft_ortho() {
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new((i as f64) * 0.5, 0.0))
            .collect();
        let out = fft(&input, None, Norm::Ortho);
        let back = ifft(&out, None, Norm::Ortho);
        assert_complex_close(&back, &input);
    }

    #[test]
    fn roundtrip_fft_ifft_forward() {
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new((i as f64) - 3.0, 0.0))
            .collect();
        let out = fft(&input, None, Norm::Forward);
        let back = ifft(&out, None, Norm::Forward);
        assert_complex_close(&back, &input);
    }

    #[test]
    fn fft_padding_roundtrip() {
        let input = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let out = fft(&input, Some(4), Norm::Backward);
        let back = ifft(&out, Some(4), Norm::Backward);
        let expected = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        assert_complex_close(&back, &expected);
    }

    #[test]
    fn fft_truncation_roundtrip() {
        let input: Vec<Complex<f64>> = (0..4).map(|i| Complex::new(i as f64, 0.0)).collect();
        let out = fft(&input, Some(2), Norm::Backward);
        let back = ifft(&out, Some(2), Norm::Backward);
        let expected = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
        assert_complex_close(&back, &expected);
    }
}
