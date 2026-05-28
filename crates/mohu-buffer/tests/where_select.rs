use mohu_buffer::{Buffer, Order, ops};
use mohu_dtype::DType;

#[test]
fn where_select_all_true() {
    let mask = Buffer::from_slice(&[1u8, 1, 1]).unwrap();

    let a = Buffer::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Buffer::from_slice(&[4i32, 5, 6]).unwrap();

    let mut dst = Buffer::alloc(DType::I32, &[3], Order::C).unwrap();

    ops::where_select::<i32>(&mask, &a, &b, &mut dst).unwrap();

    assert_eq!(dst.as_slice::<i32>().unwrap(), &[1, 2, 3]);
}

#[test]
fn where_select_all_false() {
    let mask = Buffer::from_slice(&[0u8, 0, 0]).unwrap();

    let a = Buffer::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Buffer::from_slice(&[4i32, 5, 6]).unwrap();

    let mut dst = Buffer::alloc(DType::I32, &[3], Order::C).unwrap();

    ops::where_select::<i32>(&mask, &a, &b, &mut dst).unwrap();

    assert_eq!(dst.as_slice::<i32>().unwrap(), &[4, 5, 6]);
}

#[test]
fn where_select_alternating_mask() {
    let mask = Buffer::from_slice(&[1u8, 0, 1, 0]).unwrap();

    let a = Buffer::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let b = Buffer::from_slice(&[5i32, 6, 7, 8]).unwrap();

    let mut dst = Buffer::alloc(DType::I32, &[4], Order::C).unwrap();

    ops::where_select::<i32>(&mask, &a, &b, &mut dst).unwrap();

    assert_eq!(dst.as_slice::<i32>().unwrap(), &[1, 6, 3, 8]);
}

#[test]
fn where_select_shape_mismatch() {
    let mask = Buffer::from_slice(&[1u8, 0, 1]).unwrap();

    let a = Buffer::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let b = Buffer::from_slice(&[5i32, 6, 7, 8]).unwrap();

    let mut dst = Buffer::alloc(DType::I32, &[4], Order::C).unwrap();

    let result = ops::where_select::<i32>(&mask, &a, &b, &mut dst);

    assert!(result.is_err());
}

#[test]
fn where_select_empty_buffers() {
    let mask = Buffer::from_slice::<u8>(&[]).unwrap();

    let a = Buffer::from_slice::<i32>(&[]).unwrap();
    let b = Buffer::from_slice::<i32>(&[]).unwrap();

    let mut dst = Buffer::alloc(DType::I32, &[0], Order::C).unwrap();

    ops::where_select::<i32>(&mask, &a, &b, &mut dst).unwrap();

    assert_eq!(dst.len(), 0);
}

#[test]
fn where_select_2d() {
    let mask = Buffer::from_slice(&[1u8, 0, 0, 1])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();

    let a = Buffer::from_slice(&[1i32, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();

    let b = Buffer::from_slice(&[5i32, 6, 7, 8])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();

    let mut dst = Buffer::alloc(DType::I32, &[2, 2], Order::C).unwrap();

    ops::where_select::<i32>(&mask, &a, &b, &mut dst).unwrap();

    assert_eq!(dst.as_slice::<i32>().unwrap(), &[1, 6, 7, 4]);
}
