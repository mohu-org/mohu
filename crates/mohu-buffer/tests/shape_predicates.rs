use mohu_buffer::Buffer;
use mohu_dtype::DType;

#[test]
fn is_scalar_shape_for_zero_dim() {
    let buf = Buffer::zeros(DType::F64, &[]).unwrap();
    assert!(buf.is_scalar_shape());
    assert!(!buf.is_vector());
    assert!(!buf.is_matrix());
    assert!(!buf.is_square());
}

#[test]
fn is_vector_for_1d() {
    let buf = Buffer::zeros(DType::F64, &[5]).unwrap();
    assert!(buf.is_vector());
    assert!(!buf.is_matrix());
    assert!(!buf.is_square());
}

#[test]
fn is_square_for_1x1_and_3x3() {
    let one = Buffer::zeros(DType::F64, &[1, 1]).unwrap();
    assert!(one.is_matrix());
    assert!(one.is_square());

    let three = Buffer::zeros(DType::F64, &[3, 3]).unwrap();
    assert!(three.is_square());
}

#[test]
fn is_square_false_for_rectangular_matrix() {
    let buf = Buffer::zeros(DType::F64, &[2, 3]).unwrap();
    assert!(buf.is_matrix());
    assert!(!buf.is_square());
}

#[test]
fn is_square_false_for_vector() {
    let buf = Buffer::zeros(DType::F64, &[4]).unwrap();
    assert!(!buf.is_square());
}
