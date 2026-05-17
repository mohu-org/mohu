use mohu_buffer::Buffer;
use mohu_dtype::DType;

#[test]
fn tril_zeros_above_diagonal() {
    let buf = Buffer::from_slice::<f64>(&[1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let lower = buf.tril(0).unwrap();
    assert_eq!(lower.get::<f64>(&[0, 0]).unwrap(), 1.0);
    assert_eq!(lower.get::<f64>(&[0, 1]).unwrap(), 0.0);
    assert_eq!(lower.get::<f64>(&[1, 0]).unwrap(), 3.0);
    assert_eq!(lower.get::<f64>(&[1, 1]).unwrap(), 4.0);
}

#[test]
fn triu_zeros_below_diagonal() {
    let buf = Buffer::from_slice::<f64>(&[1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let upper = buf.triu(0).unwrap();
    assert_eq!(upper.get::<f64>(&[0, 0]).unwrap(), 1.0);
    assert_eq!(upper.get::<f64>(&[0, 1]).unwrap(), 2.0);
    assert_eq!(upper.get::<f64>(&[1, 0]).unwrap(), 0.0);
    assert_eq!(upper.get::<f64>(&[1, 1]).unwrap(), 4.0);
}

#[test]
fn tril_i32_preserves_dtype() {
    let buf = Buffer::from_slice::<i32>(&[1, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let lower = buf.tril(0).unwrap();
    assert_eq!(lower.dtype(), DType::I32);
}
