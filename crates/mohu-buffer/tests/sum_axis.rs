use mohu_buffer::Buffer;
use mohu_dtype::DType;

#[test]
fn sum_axis_matches_manual_2d() {
    let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[2, 3]).unwrap();

    let axis0 = buf.sum_axis(0, false).unwrap();
    assert_eq!(axis0.shape(), &[3]);
    assert_eq!(axis0.to_vec::<f64>().unwrap(), vec![5.0, 7.0, 9.0]);

    let axis1 = buf.sum_axis(1, false).unwrap();
    assert_eq!(axis1.shape(), &[2]);
    assert_eq!(axis1.to_vec::<f64>().unwrap(), vec![6.0, 15.0]);
}

#[test]
fn sum_axis_keepdims_preserves_axis() {
    let buf = Buffer::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let summed = buf.sum_axis(0, true).unwrap();
    assert_eq!(summed.shape(), &[1, 2]);
    assert_eq!(summed.to_vec::<f64>().unwrap(), vec![4.0, 6.0]);
}

#[test]
fn sum_axis_partial_sums_equal_total() {
    let buf = Buffer::arange(0.0, 12.0, 1.0, DType::F64).unwrap().reshape(&[3, 4]).unwrap();
    let total = buf.sum_all_f64().unwrap();
    for axis in 0..buf.ndim() {
        let partial = buf.sum_axis(axis, false).unwrap().sum_all_f64().unwrap();
        assert!((partial - total).abs() < 1e-9);
    }
}
