use mohu_buffer::Buffer;

#[test]
fn allclose_broadcasts_scalar() {
    let a = Buffer::from_slice::<f64>(&[5.0, 5.0]).unwrap();
    let b = Buffer::from_slice::<f64>(&[5.0]).unwrap();
    assert!(a.allclose(&b, 0.0, 0.0).unwrap());
}

#[test]
fn allclose_same_shape_still_works() {
    let a = Buffer::from_slice::<f64>(&[1.0, 2.0]).unwrap();
    let b = Buffer::from_slice::<f64>(&[1.0, 2.0]).unwrap();
    assert!(a.allclose(&b, 0.0, 0.0).unwrap());
}

#[test]
fn allclose_incompatible_shapes_returns_broadcast_error() {
    let a = Buffer::from_slice::<f64>(&[1.0, 2.0]).unwrap();
    let b = Buffer::from_slice::<f64>(&[1.0, 2.0, 3.0]).unwrap();
    assert!(a.allclose(&b, 0.0, 0.0).is_err());
}
