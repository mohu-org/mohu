use mohu_buffer::{Buffer, ops::clip};

#[test]
fn values_within_range_unchanged() {
    let src = Buffer::from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let mut dst = Buffer::from_slice(&[0.0f32, 0.0, 0.0]).unwrap();

    clip(&src, &mut dst, 1.0f32, 3.0f32).unwrap();

    let expected = Buffer::from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_eq!(dst, expected);
}

#[test]
fn values_below_min_clamped() {
    let src = Buffer::from_slice(&[-5.0f32, 0.0, 2.0]).unwrap();
    let mut dst = Buffer::from_slice(&[0.0f32, 0.0, 0.0]).unwrap();

    clip(&src, &mut dst, 1.0f32, 5.0f32).unwrap();

    let expected = Buffer::from_slice(&[1.0f32, 1.0, 2.0]).unwrap();
    assert_eq!(dst, expected);
}

#[test]
fn values_above_max_clamped() {
    let src = Buffer::from_slice(&[1.0f32, 8.0, 10.0]).unwrap();
    let mut dst = Buffer::from_slice(&[0.0f32, 0.0, 0.0]).unwrap();

    clip(&src, &mut dst, 1.0f32, 5.0f32).unwrap();

    let expected = Buffer::from_slice(&[1.0f32, 5.0, 5.0]).unwrap();
    assert_eq!(dst, expected);
}

#[test]
fn mixed_values_clamped() {
    let src = Buffer::from_slice(&[-5.0f32, 2.0, 10.0]).unwrap();
    let mut dst = Buffer::from_slice(&[0.0f32, 0.0, 0.0]).unwrap();

    clip(&src, &mut dst, 1.0f32, 5.0f32).unwrap();

    let expected = Buffer::from_slice(&[1.0f32, 2.0, 5.0]).unwrap();
    assert_eq!(dst, expected);
}

#[test]
fn min_equals_max_constant_buffer() {
    let src = Buffer::from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let mut dst = Buffer::from_slice(&[0.0f32, 0.0, 0.0]).unwrap();

    clip(&src, &mut dst, 2.0f32, 2.0f32).unwrap();

    let expected = Buffer::from_slice(&[2.0f32, 2.0, 2.0]).unwrap();
    assert_eq!(dst, expected);
}
#[test]
fn clip_f64() {
    let src = Buffer::from_slice(&[-5.0f64, 2.0, 10.0]).unwrap();
    let mut dst = Buffer::from_slice(&[0.0f64, 0.0, 0.0]).unwrap();

    clip(&src, &mut dst, 1.0f64, 5.0f64).unwrap();

    let expected = Buffer::from_slice(&[1.0f64, 2.0, 5.0]).unwrap();
    assert_eq!(dst, expected);
}

#[test]
fn clip_i32() {
    let src = Buffer::from_slice(&[-5i32, 2, 10]).unwrap();
    let mut dst = Buffer::from_slice(&[0i32, 0, 0]).unwrap();

    clip(&src, &mut dst, 1i32, 5i32).unwrap();

    let expected = Buffer::from_slice(&[1i32, 2, 5]).unwrap();
    assert_eq!(dst, expected);
}

#[test]
fn clip_i64() {
    let src = Buffer::from_slice(&[-5i64, 2, 10]).unwrap();
    let mut dst = Buffer::from_slice(&[0i64, 0, 0]).unwrap();

    clip(&src, &mut dst, 1i64, 5i64).unwrap();

    let expected = Buffer::from_slice(&[1i64, 2, 5]).unwrap();
    assert_eq!(dst, expected);
}
#[test]
fn clip_2d_buffer() {
    let data: Vec<f32> = vec![
        -5.0, 0.0, 2.0, 10.0, 1.0, 3.0, 8.0, -2.0, 4.0, 5.0, 6.0, 7.0,
    ];

    let src = Buffer::from_slice(&data).unwrap().reshape(&[3, 4]).unwrap();

    let mut dst = Buffer::zeros(mohu_dtype::DType::F32, &[3, 4]).unwrap();

    clip(&src, &mut dst, 1.0f32, 5.0f32).unwrap();

    assert_eq!(dst.get::<f32>(&[0, 0]).unwrap(), 1.0);
    assert_eq!(dst.get::<f32>(&[0, 1]).unwrap(), 1.0);
    assert_eq!(dst.get::<f32>(&[0, 2]).unwrap(), 2.0);
    assert_eq!(dst.get::<f32>(&[0, 3]).unwrap(), 5.0);

    assert_eq!(dst.get::<f32>(&[1, 0]).unwrap(), 1.0);
    assert_eq!(dst.get::<f32>(&[1, 1]).unwrap(), 3.0);
    assert_eq!(dst.get::<f32>(&[1, 2]).unwrap(), 5.0);
    assert_eq!(dst.get::<f32>(&[1, 3]).unwrap(), 1.0);

    assert_eq!(dst.get::<f32>(&[2, 0]).unwrap(), 4.0);
    assert_eq!(dst.get::<f32>(&[2, 1]).unwrap(), 5.0);
    assert_eq!(dst.get::<f32>(&[2, 2]).unwrap(), 5.0);
    assert_eq!(dst.get::<f32>(&[2, 3]).unwrap(), 5.0);
}
