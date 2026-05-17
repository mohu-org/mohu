use mohu_buffer::Buffer;
use mohu_dtype::DType;

#[test]
fn arange_i64_large_range_has_exact_last_element() {
    let n = 100_000usize;
    let stop = n as f64;
    let buf = Buffer::arange(0.0, stop, 1.0, DType::I64).unwrap();
    assert_eq!(buf.len(), n);
    assert_eq!(buf.get::<i64>(&[n - 1]).unwrap(), (n - 1) as i64);
    assert_eq!(buf.get::<i64>(&[0]).unwrap(), 0);
}

#[test]
fn arange_i32_step_two() {
    let buf = Buffer::arange(0.0, 10.0, 2.0, DType::I32).unwrap();
    assert_eq!(buf.to_vec::<i32>().unwrap(), vec![0, 2, 4, 6, 8]);
}
