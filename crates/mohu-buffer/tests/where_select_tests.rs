use mohu_buffer::{Buffer, ops::where_select};
use mohu_dtype::DType;
use mohu_error::MohuError;
#[test]
fn basic_selection() {
    let mask = Buffer::from_slice(&[1u8, 0, 1]).unwrap();
    let a = Buffer::from_slice(&[10i32, 20, 30]).unwrap();
    let b = Buffer::from_slice(&[1i32, 2, 3]).unwrap();
    let mut d = Buffer::zeros(DType::I32, &[3]).unwrap();
    where_select::<i32>(&mask, &a, &b, &mut d).unwrap();
    assert_eq!(d.as_slice::<i32>().unwrap(), &[10, 2, 30]);
}
#[test]
fn empty_and_2d_selection() {
    let m = Buffer::from_slice(&[1u8, 0, 0, 1])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let a = Buffer::from_slice(&[10i32, 20, 30, 40])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let b = Buffer::from_slice(&[1i32, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let mut d = Buffer::zeros(DType::I32, &[2, 2]).unwrap();
    where_select::<i32>(&m, &a, &b, &mut d).unwrap();
    assert_eq!(d.as_slice::<i32>().unwrap(), &[10, 2, 3, 40]);
    let m = Buffer::zeros(DType::U8, &[0]).unwrap();
    let a = Buffer::zeros(DType::I32, &[0]).unwrap();
    let b = Buffer::zeros(DType::I32, &[0]).unwrap();
    let mut d = Buffer::zeros(DType::I32, &[0]).unwrap();
    where_select::<i32>(&m, &a, &b, &mut d).unwrap();
}
#[test]
fn rejects_shape_and_wrong_dst_dtype() {
    let m = Buffer::from_slice(&[1u8, 0]).unwrap();
    let a = Buffer::from_slice(&[1i32, 2]).unwrap();
    let b = Buffer::from_slice(&[3i32]).unwrap();
    let mut d = Buffer::zeros(DType::I32, &[2]).unwrap();
    assert!(matches!(
        where_select::<i32>(&m, &a, &b, &mut d),
        Err(MohuError::ShapeMismatch { .. })
    ));
    let mut d = Buffer::zeros(DType::F64, &[2]).unwrap();
    let b = Buffer::from_slice(&[3i32, 4]).unwrap();
    assert!(matches!(
        where_select::<i32>(&m, &a, &b, &mut d),
        Err(MohuError::DTypeMismatch { .. })
    ));
}
#[test]
fn rejects_noncontiguous_mask() {
    let m = Buffer::from_slice(&[1u8, 0, 1, 0])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap()
        .transpose();
    let a = Buffer::from_slice(&[1i32, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let b = Buffer::from_slice(&[5i32, 6, 7, 8])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let mut d = Buffer::zeros(DType::I32, &[2, 2]).unwrap();
    assert!(matches!(
        where_select::<i32>(&m, &a, &b, &mut d),
        Err(MohuError::NonContiguous)
    ));
}
