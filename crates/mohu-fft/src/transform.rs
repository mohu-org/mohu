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
	use num_complex::Complex;

	#[test]
	fn roundtrip_fft_ifft() {
		let input: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
		let out = fft(&input, None, Norm::Backward);
		let back = ifft(&out, None, Norm::Backward);
		for (a, b) in input.iter().zip(back.iter()) {
			assert!((a.re - b.re).abs() < 1e-9);
			assert!((a.im - b.im).abs() < 1e-9);
		}
	}
}
