use mohu_buffer::{ops, Buffer};
use mohu_dtype::DType;
use mohu_error::MohuError;

fn mask_buf(values: &[u8]) -> Buffer {
    Buffer::from_slice(values).unwrap()
}

fn f64_buf(values: &[f64]) -> Buffer {
    Buffer::from_slice(values).unwrap()
}

#[test]
fn all_true_selects_from_a() {
    let mask = mask_buf(&[1, 1, 1]);
    let a = f64_buf(&[1.0, 2.0, 3.0]);
    let b = f64_buf(&[9.0, 9.0, 9.0]);
    let mut dst = Buffer::zeros(DType::F64, &[3]).unwrap();

    ops::where_select::<f64>(&mask, &a, &b, &mut dst).unwrap();

    assert_eq!(dst.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn all_false_selects_from_b() {
    let mask = mask_buf(&[0, 0, 0]);
    let a = f64_buf(&[1.0, 2.0, 3.0]);
    let b = f64_buf(&[4.0, 5.0, 6.0]);
    let mut dst = Buffer::zeros(DType::F64, &[3]).unwrap();

    ops::where_select::<f64>(&mask, &a, &b, &mut dst).unwrap();

    assert_eq!(dst.as_slice::<f64>().unwrap(), &[4.0, 5.0, 6.0]);
}

#[test]
fn alternating_mask() {
    let mask = mask_buf(&[1, 0, 1, 0]);
    let a = f64_buf(&[10.0, 20.0, 30.0, 40.0]);
    let b = f64_buf(&[1.0, 2.0, 3.0, 4.0]);
    let mut dst = Buffer::zeros(DType::F64, &[4]).unwrap();

    ops::where_select::<f64>(&mask, &a, &b, &mut dst).unwrap();

    assert_eq!(dst.as_slice::<f64>().unwrap(), &[10.0, 2.0, 30.0, 4.0]);
}

#[test]
fn shape_mismatch_returns_error() {
    let mask = mask_buf(&[1, 0]);
    let a = f64_buf(&[1.0, 2.0, 3.0]);
    let b = f64_buf(&[4.0, 5.0, 6.0]);
    let mut dst = Buffer::zeros(DType::F64, &[2]).unwrap();

    let err = ops::where_select::<f64>(&mask, &a, &b, &mut dst).unwrap_err();
    assert!(matches!(err, MohuError::ShapeMismatch { .. }));
}

#[test]
fn empty_buffers_produce_empty_output() {
    let mask = mask_buf(&[]);
    let a = f64_buf(&[]);
    let b = f64_buf(&[]);
    let mut dst = Buffer::zeros(DType::F64, &[0]).unwrap();

    ops::where_select::<f64>(&mask, &a, &b, &mut dst).unwrap();

    assert!(dst.is_empty());
    assert_eq!(dst.as_slice::<f64>().unwrap().len(), 0);
}

#[test]
fn two_d_mask_and_buffers() {
    let mask = mask_buf(&[1, 0, 0, 1]).reshape(&[2, 2]).unwrap();
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    let b = f64_buf(&[5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]).unwrap();
    let mut dst = Buffer::zeros(DType::F64, &[2, 2]).unwrap();

    ops::where_select::<f64>(&mask, &a, &b, &mut dst).unwrap();

    assert_eq!(dst.as_slice::<f64>().unwrap(), &[1.0, 6.0, 7.0, 4.0]);
}
