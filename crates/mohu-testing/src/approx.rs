// approx — element-wise approximate equality helpers

pub fn assert_allclose(a: &[f32], b: &[f32]) {
    assert_allclose_tol(a, b, 1e-6, 1e-6);
}

pub fn assert_allclose_tol(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
    assert!(rtol >= 0.0, "rtol must be non-negative, got {}", rtol);
    assert!(atol >= 0.0, "atol must be non-negative, got {}", atol);
    assert_eq!(a.len(), b.len(), "Lengths differ");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        // Handle infinities: equal infinities (same sign) are considered close.
        if x.is_infinite() || y.is_infinite() {
            assert!(x == y, "Values differ at index {}: {} vs {}", i, x, y);
            continue;
        }
        // NaN is never close to anything, including itself.
        if x.is_nan() || y.is_nan() {
            panic!("NaN encountered at index {}: {} vs {}", i, x, y);
        }
        let diff = (x - y).abs();
        let threshold = atol + rtol * y.abs();
        assert!(
            diff <= threshold,
            "Values differ at index {}: {} vs {} (diff={}, threshold={})",
            i,
            x,
            y,
            diff,
            threshold
        );
    }
}
