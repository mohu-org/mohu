/// Assert that two equal-length float slices are close within tolerances.
///
/// # Example
///
/// ```rust,ignore
/// use mohu_testing::assert_allclose;
///
/// assert_allclose!(vec![1.0, 2.0], vec![1.0, 2.0000001], atol = 1e-6);
/// ```
#[macro_export]
macro_rules! assert_allclose {
	($actual:expr, $expected:expr, atol = $atol:expr $(,)?) => {
		$crate::assert_allclose!($actual, $expected, rtol = 0.0, atol = $atol)
	};
	($actual:expr, $expected:expr, rtol = $rtol:expr, atol = $atol:expr $(,)?) => {{
		let actual_value = $actual;
		let expected_value = $expected;
		let actual = ::core::convert::AsRef::<[_]>::as_ref(&actual_value);
		let expected = ::core::convert::AsRef::<[_]>::as_ref(&expected_value);

		assert_eq!(
			actual.len(),
			expected.len(),
			"length mismatch: left = {}, right = {}",
			actual.len(),
			expected.len()
		);

		for (index, (&lhs, &rhs)) in actual.iter().zip(expected.iter()).enumerate() {
			let difference = (lhs - rhs).abs();
			let tolerance = $atol + $rtol * lhs.abs().max(rhs.abs());
			assert!(
				difference <= tolerance,
				"values differ at index {index}: left = {:?}, right = {:?}, diff = {:?}, tolerance = {:?}",
				lhs,
				rhs,
				difference,
				tolerance
			);
		}
	}};
}
