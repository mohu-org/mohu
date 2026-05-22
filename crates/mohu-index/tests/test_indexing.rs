use mohu_buffer::Buffer;
use mohu_dtype::dtype::DType;
use mohu_index::{index_bool, index_take, MohuError};

#[test]
fn test_index_bool() {
    let src = Buffer::from_slice(&[10i32, 20, 30, 40]).unwrap();
    let mask = Buffer::from_slice(&[true, false, true, false]).unwrap();
    
    let result = index_bool(&src, &mask).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice::<i32>().unwrap(), &[10, 30]);
}

#[test]
fn test_index_bool_errors() {
    let src = Buffer::from_slice(&[10i32, 20, 30, 40]).unwrap();
    
    // DomainError for non-bool mask
    let mask = Buffer::from_slice(&[1i32, 0, 1, 0]).unwrap();
    assert!(matches!(index_bool(&src, &mask), Err(MohuError::DomainError { .. })));

    // ShapeMismatch
    let mask2 = Buffer::from_slice(&[true, false]).unwrap();
    assert!(matches!(index_bool(&src, &mask2), Err(MohuError::ShapeMismatch { .. })));
}

#[test]
fn test_index_take() {
    // 2D src: shape (2, 3)
    let src = Buffer::from_slice_2d(&[
        &[1i32, 2, 3][..],
        &[4, 5, 6][..],
    ]).unwrap();

    // indices: shape (2,)
    let indices = Buffer::from_slice(&[1i64, 0]).unwrap();

    // take along axis 0
    let result = index_take(&src, &indices, 0).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    let slice = result.as_slice::<i32>().unwrap();
    assert_eq!(slice, &[4, 5, 6, 1, 2, 3]);

    // take along axis 1
    let result2 = index_take(&src, &indices, 1).unwrap();
    assert_eq!(result2.shape(), &[2, 2]);
    let slice2 = result2.as_slice::<i32>().unwrap();
    // expected: row 0: indices 1, 0 -> 2, 1
    //           row 1: indices 1, 0 -> 5, 4
    assert_eq!(slice2, &[2, 1, 5, 4]);
}

#[test]
fn test_index_take_errors() {
    let src = Buffer::from_slice(&[10i32, 20, 30]).unwrap();
    
    let wrong_dtype = Buffer::from_slice(&[1i32, 0]).unwrap();
    assert!(matches!(index_take(&src, &wrong_dtype, 0), Err(MohuError::DomainError { .. })));

    let indices = Buffer::from_slice(&[1i64, 3]).unwrap();
    assert!(matches!(index_take(&src, &indices, 0), Err(MohuError::IndexOutOfBounds { .. })));

    let indices_neg = Buffer::from_slice(&[1i64, -1]).unwrap();
    assert!(matches!(index_take(&src, &indices_neg, 0), Err(MohuError::IndexOutOfBounds { .. })));

    let valid_indices = Buffer::from_slice(&[1i64, 0]).unwrap();
    assert!(matches!(index_take(&src, &valid_indices, 1), Err(MohuError::AxisOutOfRange { .. })));
}
