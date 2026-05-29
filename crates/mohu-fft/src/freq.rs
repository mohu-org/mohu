/// Frequency-axis helpers similar to NumPy's `fftfreq` and `fftshift`.

/// Return the Discrete Fourier Transform sample frequencies for a window of
/// length `n` and sample spacing `d` (default 1.0).
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
	if n == 0 {
		return Vec::new();
	}
	let val = 1.0 / (n as f64 * d);
	let mut freqs = Vec::with_capacity(n);
	// For even `n` the Nyquist frequency (n/2) should be negative (-0.5/d).
	// Use `pos_len = (n + 1) / 2` as the number of non-negative frequency bins.
	let pos_len = (n + 1) / 2;
	for i in 0..pos_len {
		freqs.push(i as f64 * val);
	}
	for i in pos_len..n {
		freqs.push(-((n - i) as f64) * val);
	}
	freqs
}

/// Alias for `fftfreq` for real-input transforms; behavior is identical.
pub fn rfftfreq(n: usize, d: f64) -> Vec<f64> {
	if n == 0 {
		return Vec::new();
	}
	let val = 1.0 / (n as f64 * d);
	let count = n / 2 + 1;
	(0..count).map(|i| i as f64 * val).collect()
}

/// Shift the zero-frequency component to the center of the spectrum.
pub fn fftshift<T: Clone>(mut v: Vec<T>) -> Vec<T> {
	let n = v.len();
	if n == 0 {
		return v;
	}
	let mid = n / 2;
	let mut out = Vec::with_capacity(n);
	out.extend_from_slice(&v[mid..]);
	out.extend_from_slice(&v[..mid]);
	out
}

/// The inverse of `fftshift`.
pub fn ifftshift<T: Clone>(mut v: Vec<T>) -> Vec<T> {
	let n = v.len();
	if n == 0 {
		return v;
	}
	let mid = (n + 1) / 2;
	let mut out = Vec::with_capacity(n);
	out.extend_from_slice(&v[mid..]);
	out.extend_from_slice(&v[..mid]);
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_fftfreq_len() {
		let f = fftfreq(4, 1.0);
		assert_eq!(f, vec![0.0, 0.25, -0.5, -0.25]);
	}

	#[test]
	fn test_fftshift() {
		let v = vec![0, 1, 2, 3];
		let s = fftshift(v.clone());
		assert_eq!(s, vec![2, 3, 0, 1]);
		let r = ifftshift(s);
		assert_eq!(r, v);
	}
}
