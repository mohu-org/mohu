use mohu_buffer::Buffer;
use mohu_dtype::dtype::DType;
use mohu_error::MohuError;
use mohu_index::{index_bool, index_take};

fn shaped<T: mohu_dtype::scalar::Scalar>(v: &[T], shape: &[usize]) -> Buffer {
    Buffer::from_slice(v).unwrap().reshape(shape).unwrap()
}

#[test]
fn bool_index_selects_owning_values() {
    let src = shaped(&[1i32, 2, 3, 4], &[2, 2]);
    let mask = shaped(&[true, false, false, true], &[2, 2]);
    let out = index_bool(&src, &mask).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.dtype(), DType::I32);
    assert_eq!(out.to_vec::<i32>().unwrap(), vec![1, 4]);
    assert!(out.is_writeable());
}

#[test]
fn bool_index_errors() {
    let src = Buffer::from_slice(&[1i32, 2]).unwrap();
    let wrong = Buffer::from_slice(&[1i32, 0]).unwrap();
    assert!(matches!(
        index_bool(&src, &wrong),
        Err(MohuError::DomainError { .. })
    ));
    let mask = Buffer::from_slice(&[true]).unwrap();
    assert!(matches!(
        index_bool(&src, &mask),
        Err(MohuError::ShapeMismatch { .. })
    ));
}

#[test]
fn take_supports_axis_and_nd_indices() {
    let src = shaped(&[10i32, 11, 12, 20, 21, 22], &[2, 3]);
    let rows = Buffer::from_slice(&[1i64, 0]).unwrap();
    let out = index_take(&src, &rows, 0).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.to_vec::<i32>().unwrap(), vec![20, 21, 22, 10, 11, 12]);
    let cols = shaped(&[2i64, 0, 1, 1], &[2, 2]);
    let out = index_take(&src, &cols, 1).unwrap();
    assert_eq!(out.shape(), &[2, 2, 2]);
    assert_eq!(
        out.to_vec::<i32>().unwrap(),
        vec![12, 10, 11, 11, 22, 20, 21, 21]
    );
}

#[test]
fn take_rejects_negative_oob_and_bad_axis() {
    let src = Buffer::from_slice(&[1i32, 2, 3]).unwrap();
    for value in [-1i64, 3] {
        let idx = Buffer::from_slice(&[value]).unwrap();
        assert!(
            matches!(index_take(&src,&idx,0),Err(MohuError::IndexOutOfBounds{index,axis:0,size:3}) if index==value)
        );
    }
    let idx = Buffer::from_slice(&[0i64]).unwrap();
    assert!(matches!(
        index_take(&src, &idx, 1),
        Err(MohuError::AxisOutOfRange { .. })
    ));
}

#[test]
fn take_empty_indices_substitute_shape() {
    let src = Buffer::from_slice::<i32>(&[])
        .unwrap()
        .reshape(&[0, 3])
        .unwrap();
    let idx = Buffer::from_slice::<i64>(&[])
        .unwrap()
        .reshape(&[0, 2])
        .unwrap();
    let out = index_take(&src, &idx, 1).unwrap();
    assert_eq!(out.shape(), &[0, 0, 2]);
    assert_eq!(out.len(), 0);
}
