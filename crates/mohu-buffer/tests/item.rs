use mohu_buffer::Buffer;
use mohu_dtype::DType;
use mohu_error::MohuError;

#[test]
fn item_extracts_scalar_from_length_one_buffer() {
    let buf = Buffer::from_slice::<f64>(&[42.0]).unwrap();
    assert_eq!(buf.item::<f64>().unwrap(), 42.0);
}

#[test]
fn item_extracts_from_zero_d_scalar() {
    let mut buf = Buffer::zeros(DType::I32, &[]).unwrap();
    buf.set::<i32>(&[], 7).unwrap();
    assert_eq!(buf.item::<i32>().unwrap(), 7);
}

#[test]
fn item_rejects_multi_element_buffer() {
    let buf = Buffer::zeros(DType::F32, &[2]).unwrap();
    assert!(matches!(buf.item::<f32>(), Err(MohuError::Internal(_))));
}
