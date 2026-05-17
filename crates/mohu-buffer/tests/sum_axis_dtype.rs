use mohu_buffer::Buffer;
use mohu_dtype::DType;

#[test]
fn sum_axis_i32_returns_i64() {
    let buf = Buffer::from_slice::<i32>(&[1, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let summed = buf.sum_axis(0, false).unwrap();
    assert_eq!(summed.dtype(), DType::I64);
    assert_eq!(summed.shape(), &[2]);
    assert_eq!(summed.to_vec::<i64>().unwrap(), vec![4, 6]);
}

#[test]
fn sum_axis_f32_returns_f32() {
    let buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let summed = buf.sum_axis(1, false).unwrap();
    assert_eq!(summed.dtype(), DType::F32);
    assert_eq!(summed.to_vec::<f32>().unwrap(), vec![3.0, 7.0]);
}
