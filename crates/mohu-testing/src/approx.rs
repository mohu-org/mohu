pub fn assert_allclose(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Lengths differ");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-6, "Values differ: {} and {}", x, y);
    }
}
